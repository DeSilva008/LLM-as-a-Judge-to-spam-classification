import pandas as pd
import numpy as np
import requests
import json
import time
import re
from tqdm import tqdm

# --- NEW IMPORTS FOR FEATURE F_LOCAL_SIM ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# SYSTEM PROMPTS (OPTIMIZED - 40% REDUCTION)
# ==========================================

# 1. F_LOC_DIST (GEO-SPATIAL DISCREPANCY) PROMPT
SYSTEM_PROMPT_LOC_DIST = """LocDist: staff, friends and neighbours boosting a nearby hotel are more biased than a foreign tourist.
Judge how close the reviewer's declared home is to the hotel — same town, same country, or abroad.
If the reviewer's location is missing, answer "insufficient"."""

# 2. F_CONFLICT (COMPETITOR BASHING) PROMPT
SYSTEM_PROMPT_CONFLICT = """Conflict: a reviewer attacking one hotel while praising a nearby competitor may have a stake in the rival.
Judge whether a low target Rating is paired with a high rating for a nearby hotel in the user's History.
If the target rating is above 2, or History has no rated nearby hotel, answer "insufficient"."""

# 3. F_LOCAL_SIM (SEMANTIC CONSENSUS) PROMPT
SYSTEM_PROMPT_LOCAL_SIM = """Analyze 'F_local_sim' (Semantic Consensus) Score:

- **NORMAL (Score > 0.20)**:
  - Meaning: Review discusses themes common to this hotel (grounded in reality).

- **ANOMALOUS (Score <= 0.20)**:
  - Meaning: Review is semantically disjoint from hotel reality.
  - Risks:
    1. **Hallucination:** Mentioning amenities that don't exist (e.g., 'snow' at beach).
    2. **Generic Filler:** Vague text ('Good place') applicable to any hotel.
    3. **Wrong Venue.**
  - Action: Mark 'is_fake_suspect': true.

Output strictly in JSON.
"""

# ==========================================
# 1. CALCULATORS (UNCHANGED)
# ==========================================

def detect_user_location_column(df: pd.DataFrame):
    """FIX: precompute_geo_scores read 'user/userLocation' while Layer4 read
    'user/userLocation/name'. If the real column is the other one, every row
    fell to the else-branch and f_loc_dist came out 0.0 for all 18,815 rows.
    Detect it instead of hardcoding."""
    prefer = ['user/userLocation/name', 'user/userLocation',
              'user/userLocation/shortName', 'userLocation']
    for c in prefer:
        if c in df.columns and df[c].notna().mean() > 0.05:
            return c
    cands = [c for c in df.columns
             if 'userlocation' in str(c).lower().replace(' ', '')]
    cands = [c for c in cands if df[c].notna().mean() > 0.05]
    return cands[0] if cands else None


def detect_hotel_location_column(df: pd.DataFrame):
    """FIX: the hotel's location was read from 'Review_1_Location', which is a
    USER-HISTORY column describing a different property entirely."""
    prefer = ['placeInfo/locationString', 'placeInfo/addressObj/city',
              'placeInfo/address', 'placeInfo/locationId', 'hotel_location']
    for c in prefer:
        if c in df.columns and df[c].notna().mean() > 0.05:
            return c
    cands = [c for c in df.columns
             if str(c).lower().startswith('placeinfo')
             and any(k in str(c).lower() for k in ('location', 'city', 'address'))]
    cands = [c for c in cands if df[c].notna().mean() > 0.05]
    return cands[0] if cands else None


def detect_hotel_region_column(df: pd.DataFrame):
    """Region/town used by f_conflict to find geographic neighbours."""
    return detect_hotel_location_column(df)


def precompute_geo_scores(df: pd.DataFrame):
    """Component 1 (Loc Dist): Calculates F_loc_dist."""
    print("--- Pre-computing F_loc_dist (Geo-Spatial) Scores ---")
    
    def parse_loc(loc_str):
        if pd.isna(loc_str) or str(loc_str).strip() == "":
            return None, None
        parts = [p.strip().lower() for p in str(loc_str).split(',')]
        if len(parts) >= 2:
            return parts[0], parts[-1] 
        elif len(parts) == 1:
            return None, parts[0]
        return None, None

    temp_df = df.copy()
    
    hotel_col = detect_hotel_location_column(df)
    print(f"    hotel location column -> {hotel_col}")
    if hotel_col:
        hotel_locs = temp_df[hotel_col].apply(parse_loc)
    else:
        print("    [!] No hotel location column found. f_loc_dist will be NaN.")
        return np.full(len(df), np.nan)

    temp_df['hotel_city'] = hotel_locs.apply(lambda x: x[0])
    temp_df['hotel_country'] = hotel_locs.apply(lambda x: x[1])

    user_col = detect_user_location_column(df)
    print(f"    user location column   -> {user_col}")
    if user_col:
        user_locs = temp_df[user_col].apply(parse_loc)
    else:
        print("    [!] No user location column found. f_loc_dist will be NaN.")
        return np.full(len(df), np.nan)
        
    temp_df['user_city'] = user_locs.apply(lambda x: x[0])
    temp_df['user_country'] = user_locs.apply(lambda x: x[1])

    conditions = [
        (temp_df['user_city'].notna()) & 
        (temp_df['hotel_city'].notna()) & 
        (temp_df['user_city'] == temp_df['hotel_city']),
        
        (temp_df['user_country'].notna()) & 
        (temp_df['hotel_country'].notna()) & 
        (temp_df['user_country'] == temp_df['hotel_country'])
    ]
    
    choices = [1.0, 0.15]
    geo_scores = np.select(conditions, choices, default=0.0)
    # FIX: rows where either side's location is unknown are NaN, not "international".
    unknown = (temp_df['user_city'].isna() & temp_df['user_country'].isna()) | \
              (temp_df['hotel_city'].isna() & temp_df['hotel_country'].isna())
    geo_scores = np.where(unknown.to_numpy(), np.nan, geo_scores)
    print("--- F_loc_dist Calculation Complete ---")
    return geo_scores

def calculate_geo_conflict_score(df, target_region=None, region_column=None):
    """Component 1 (Conflict): Calculates F_conflict."""
    print(f"--- Pre-computing F_conflict Scores for Region: {target_region} ---")

    def check_row_conflict(row):
        # FIX: region was hardcoded to "Kalutara" for every hotel in a 17-hotel
        # dataset, so this feature was 99.9% zero.
        region = row.get(region_column) if region_column else target_region
        if region is None or pd.isna(region) or str(region).strip() == "":
            return np.nan
        region = str(region)

        try:
            current_rating = float(row.get('rating', np.nan))
        except:
            current_rating = np.nan
        # FIX: was `float(row.get('rating', 5))` -- a missing rating defaulted to
        # 5 and silently scored as safe.
        if pd.isna(current_rating):
            return np.nan

        if current_rating > 3:
            return 0.0
        
        found_neighbor_high_rating = False
        for i in range(1, 11): 
            loc_col = f'Review_{i}_Location'
            rating_col = f'Review_{i}_Rating'
            
            if loc_col in row and rating_col in row:
                hist_loc = str(row[loc_col])
                try:
                    if pd.isna(row[rating_col]) or str(row[rating_col]).lower() == 'nan':
                        continue
                    hist_rating = float(row[rating_col])
                except:
                    continue 
                
                is_neighbor = region.lower() in hist_loc.lower()
                is_high_rating = hist_rating >= 4
                
                if is_neighbor and is_high_rating:
                    found_neighbor_high_rating = True
                    break
        
        if found_neighbor_high_rating:
            return 1.0
        return 0.0

    return df.apply(check_row_conflict, axis=1)

def precompute_local_consensus_features(df: pd.DataFrame, text_column='text', date_column='publishedDate'):
    """Component 1 (Local Sim): Calculates F_local_sim scores."""
    print("--- Pre-computing F_local_sim Scores ---")
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    scores = []
    context_strs = []
    
    for target_index in tqdm(df.index, desc="Calculating TF-IDF per row"):
        
        target_text = str(df.loc[target_index, text_column])
        word_count = len(target_text.split())
        
        if word_count < 5:
            scores.append(1.0)
            context_strs.append("Review too short for semantic analysis.")
            continue

        target_date = df.loc[target_index, date_column]
        if pd.isnull(target_date):
            window_df = df 
        else:
            start_date = target_date - pd.Timedelta(days=90)
            window_df = df[
                (df[date_column] >= start_date) & 
                (df[date_column] <= target_date) & 
                (df.index != target_index)
            ]
        
        if window_df.empty:
            scores.append(0.5)
            context_strs.append("Insufficient historical context.")
            continue

        corpus = window_df[text_column].fillna('').tolist() + [target_text]
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            vec_r = tfidf_matrix[-1] 
            vec_history = tfidf_matrix[:-1] 
            
            vec_mu_hotel = np.mean(vec_history, axis=0)
            vec_mu_hotel_arr = np.asarray(vec_mu_hotel)
            
            score = cosine_similarity(vec_r, vec_mu_hotel_arr)[0][0]
            scores.append(float(score))
            
            flat_centroid = np.asarray(vec_mu_hotel).flatten()
            sorted_indices = flat_centroid.argsort()[::-1][:5]
            feature_names = np.array(vectorizer.get_feature_names_out())
            top_keywords = feature_names[sorted_indices]
            
            context_strs.append(f"Hotel Centroid Themes (90-day): {', '.join(top_keywords)}")
            
        except ValueError:
            scores.append(0.0)
            context_strs.append("Error in vectorization.")

    print("--- F_local_sim Calculation Complete ---")
    return pd.DataFrame({'Sim_Score': scores, 'Sim_Context_Str': context_strs})

# ==========================================
# 2. PROMPT VARIABLES (UNCHANGED)
# ==========================================

def prepare_loc_dist_variables(row, loc_score):
    """Component 2 (Loc Dist)"""
    target_review = str(row.get('text', ''))
    if target_review == 'nan' or not target_review.strip():
        target_review = "[NO TEXT CONTENT]"

    user_name = row.get('user/name', 'Unknown User')
    user_loc = row.get('user/userLocation', 'Unknown Location')
    total_contribs = row.get('user/contributions/totalContributions', 0)
    
    history_snippets = []
    for i in range(2, 4): 
        hist_text = str(row.get(f'Review_{i}_Text', ''))
        hist_rating = str(row.get(f'Review_{i}_Rating', 'N/A'))
        if hist_text and hist_text.lower() != 'nan':
            history_snippets.append(f"- Review {i} (Rating {hist_rating}): \"{hist_text[:100]}...\"")
            
    history_str = "\n".join(history_snippets) if history_snippets else "No prior history available."
    
    context_variable = (
        f"User Identity: {user_name}\n"
        f"Declared Location: {user_loc}\n"
        f"Account Maturity: {total_contribs} total contributions.\n"
        f"Previous Review History:\n{history_str}"
    )
    calculated_score = f"{float(loc_score):.2f}"
    return {"target_review": target_review, "context_variable": context_variable, "calculated_score": calculated_score}

def prepare_conflict_variables(row, score):
    """Component 2 (Conflict)"""
    target_review = str(row.get('text', ''))
    if target_review == 'nan' or not target_review.strip():
        target_review = "[No Text Content]"

    history_entries = []
    for i in range(1, 6): 
        r_loc = row.get(f'Review_{i}_Location')
        r_rating = row.get(f'Review_{i}_Rating')
        r_text = str(row.get(f'Review_{i}_Text', ''))[:100]
        if pd.notna(r_loc) and pd.notna(r_rating):
            history_entries.append(f"- Review {i}: Rated {r_rating}/5 at {r_loc}. Excerpt: \"{r_text}...\"")
    
    if not history_entries:
        context_variable = "User has no visible history."
    else:
        context_variable = "User History:\n" + "\n".join(history_entries)

    score_label = "HIGH RISK (Competitor Conflict)" if float(score) == 1.0 else "Low Risk"
    calculated_score_str = f"{float(score):.2f} - {score_label}"
    return {"target_review": target_review, "context_variable": context_variable, "calculated_score": calculated_score_str}

def prepare_local_sim_variables(row):
    """Component 2 (Local Sim)"""
    target_review = str(row.get('text', ''))
    context_variable = str(row.get('Sim_Context_Str', 'No Context Available'))
    raw_score = row.get('Sim_Score', 0.0)
    formatted_score = f"{float(raw_score):.3f}"

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": formatted_score
    }


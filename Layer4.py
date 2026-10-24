import pandas as pd
import numpy as np
import requests
import json
import time
import re
from tqdm import tqdm
from dateutil import parser
from datetime import timedelta

# Defined based on Input File context
TARGET_HOTEL_NAME = "Mermaid Hotel & Club" 

# ==========================================
# SYSTEM PROMPTS (OPTIMIZED - 40% REDUCTION)
# ==========================================

# 1. F_WEIGHT_DEV (WEIGHTED EARLY BIAS) PROMPT
SYSTEM_PROMPT_WEIGHTED_BIAS = """WeightDev: an attack lands hardest when a hotel has few reviews; the same rating is harmless once there are thousands.
Judge the target Rating's deviation from HotelAverageRating, weighted by HotelTotalReviews.
If HotelTotalReviews is missing, answer "insufficient"."""

# 2. F_HIT_EXTREMIST (ONE-HIT EXTREMIST) PROMPT
SYSTEM_PROMPT_ONE_HIT = """OneHit: accounts created for a single mission post one review and vanish.
Judge whether TotalContributions of 1 is paired with anything else unusual — a brand-new account, or an extreme rating.
If TotalContributions is missing, answer "insufficient"."""

# 3. F_REPEAT_RAPID (RAPID REPEATED PRODUCT) PROMPT
SYSTEM_PROMPT_REPEAT_RAPID = """Repeat: genuine guests rarely review the same hotel twice in quick succession.
Judge whether any hotel appears more than once in the user's History, excluding the review being judged.
If History has under 2 entries, answer "insufficient"."""

# 4. F_FRESH (FRESHNESS/MEMORY DECAY) PROMPT
SYSTEM_PROMPT_FRESHNESS = """Fresh: a review written long after the stay relies on faded memory and is less reliable.
Judge the delay between the travel date and the publish date.
If either date is missing, answer "insufficient"."""

# ==========================================
# 1. CALCULATORS (UNCHANGED)
# ==========================================

def calculate_weighted_early_bias(df, target_col='rating', date_col='publishedDate', mu=4.6):
    """Component 1 (Weighted Bias): Calculates the Weighted Early Bias score."""
    print("--- Pre-computing Weighted Early Bias Scores ---")
    df_calc = df.copy()
    if target_col not in df_calc.columns:
        return [0.0] * len(df)
        
    df_calc[date_col] = pd.to_datetime(df_calc[date_col], errors='coerce')
    df_calc['orig_index'] = df_calc.index
    df_sorted = df_calc.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    df_sorted['rank_r'] = df_sorted.index + 1
    
    safe_ratings = df_sorted[target_col].fillna(mu)
    numerator = np.abs(safe_ratings - mu)
    denominator = np.sqrt(df_sorted['rank_r'])
    
    df_sorted['f_weight_dev'] = numerator / denominator
    result_series = df_sorted.set_index('orig_index')['f_weight_dev'].sort_index()
    return result_series

def calculate_one_hit_extremist_score(df: pd.DataFrame) -> pd.Series:
    """Component 1 (One-Hit Extremist): Calculates F_hit_extremist."""
    print("--- Pre-computing One-Hit Extremist Scores ---")
    col_contribs = 'user/contributions/totalContributions'
    col_rating = 'rating'
    
    if col_contribs not in df.columns:
        return pd.Series([0] * len(df))

    contribs_numeric = pd.to_numeric(df[col_contribs], errors='coerce').fillna(0)
    ratings_numeric = pd.to_numeric(df[col_rating], errors='coerce')
    
    is_one_hit = (contribs_numeric == 1)
    is_extreme = ratings_numeric.isin([1, 5, 1.0, 5.0])
    
    return (is_one_hit & is_extreme).astype(int)

def calculate_f_repeat_rapid(row, target_hotel_name, lookback_window=180):
    """Component 1 (Rapid Repeat): Checks for velocity."""
    try:
        current_date_str = str(row.get('publishedDate', ''))
        if not current_date_str or current_date_str.lower() == 'nan':
            return 0 
        current_date = parser.parse(current_date_str)

        history_indices = [1, 2, 3, 4] 
        for i in history_indices:
            hist_hotel_col = f'Review_{i}_Target_Hotel'
            hist_date_col = f'Review_{i}_Date'
            
            if hist_hotel_col not in row or hist_date_col not in row:
                continue
            hist_hotel = str(row[hist_hotel_col])
            hist_date_str = str(row[hist_date_col])
            
            if pd.isna(hist_hotel) or pd.isna(hist_date_str) or hist_hotel.lower() == 'nan':
                continue

            if hist_hotel.strip().lower() == target_hotel_name.strip().lower():
                try:
                    hist_date = parser.parse(hist_date_str)
                    time_diff = abs((current_date - hist_date).days)
                    # FIX: the target review is present in its own history columns
                    # (57% of Review_1 entries match the target hotel). A 0-day gap
                    # is the review matching itself, not a repeat visit. That
                    # self-match drove this feature to 81% positive.
                    if (hist_date.year == current_date.year
                            and hist_date.month == current_date.month):
                        continue   # month-precision history can't distinguish
                                   # a self-match from a genuine same-month repeat
                    if time_diff < lookback_window:
                        return 1 
                except (ValueError, TypeError):
                    continue 
        return 0 
    except Exception as e:
        return 0

def calculate_freshness_decay(df: pd.DataFrame, pub_col='publishedDate', travel_col='travelDate') -> pd.Series:
    """Component 1 (Freshness Decay) [NEW]: Calculates F_fresh score."""
    print("--- Pre-computing Freshness Decay Scores ---")
    
    if travel_col not in df.columns:
        print(f"Warning: {travel_col} not found. Returning 0 for Freshness Decay.")
        return pd.Series([0.0] * len(df))

    dt_pub = pd.to_datetime(df[pub_col], errors='coerce')
    dt_travel = pd.to_datetime(df[travel_col], errors='coerce')

    lag_days = (dt_pub - dt_travel).dt.days.abs()
    # FIX: lambda lowered from 0.015 to 0.002. At 0.015 the score hit 0.9 after
    # 154 days, so a 6-month lag and a 5-year lag were indistinguishable, and the
    # rubric's "HISTORICAL (>0.9): years ago" was wrong by an order of magnitude.
    f_fresh = 1 - np.exp(-0.002 * lag_days)
    # FIX: was fillna(0.0), which scored a missing travel date as perfectly fresh.
    return f_fresh

# ==========================================
# 2. PROMPT VARIABLES (UNCHANGED)
# ==========================================

def prepare_weighted_bias_variables(row, calculated_score):
    """Component 2 (Weighted Bias)"""
    title = str(row.get('title', '')).strip() if pd.notna(row.get('title')) else ""
    text = str(row.get('text', '')).strip() if pd.notna(row.get('text')) else ""
    target_review = f"Title: {title}\nBody: {text}"

    history_reviews = []
    for i in range(1, 6):
        hist_col = f'Review_{i}_Text'
        if hist_col in row.index and pd.notna(row[hist_col]):
            clean_text = str(row[hist_col]).replace("...", "").strip()
            if len(clean_text) > 10: 
                history_reviews.append(f"[History {i}]: {clean_text}")
    
    context_variable = "\n".join(history_reviews) if history_reviews else "No historical reviews available."
    metric_display = f"{float(calculated_score):.4f}"

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": metric_display
    }

def prepare_one_hit_variables(row, f_score_value):
    """Component 2 (One-Hit Extremist)"""
    raw_text = row.get('text', '')
    target_review = "[NO TEXT CONTENT - RATING ONLY]" if pd.isna(raw_text) or str(raw_text).strip() == '' else str(raw_text).strip()
    
    user_name = row.get('user/name', 'Anonymous')
    total_contribs = row.get('user/contributions/totalContributions', 'Unknown')
    rating_given = row.get('rating', 'Unknown')
    
    context_variable = (
        f"User Identity: '{user_name}'.\n"
        f"Platform History: This user has exactly {total_contribs} lifetime contribution(s).\n"
        f"Current Action: They left a rating of {rating_given}/5."
    )
    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": str(int(f_score_value))
    }

def prepare_repeat_rapid_variables(row, f_score):
    """Component 2 (Rapid Repeat)"""
    target_review_text = str(row.get('text', ''))
    if target_review_text.lower() == 'nan':
        target_review_text = "[No Text Content Available]"

    history_entries = []
    for i in [1, 2, 3, 4]:
        h_hotel = row.get(f'Review_{i}_Target_Hotel')
        h_date = row.get(f'Review_{i}_Date')
        h_text = row.get(f'Review_{i}_Text')
        h_rating = row.get(f'Review_{i}_Rating')
        
        if pd.notna(h_hotel) and pd.notna(h_text):
            clean_text = str(h_text).replace('\n', ' ')[:200] 
            entry = f"- Review {i} ({h_date}): Rated {h_rating}/5 for '{h_hotel}'. Snippet: \"{clean_text}...\""
            history_entries.append(entry)

    context_variable = "User History:\n" + "\n".join(history_entries) if history_entries else "User History: No previous reviews found."
    metric_str = "1 (HIGH RISK: Repeat review < 180 days)" if f_score == 1 else "0 (Normal Velocity)"

    return {
        "target_review": target_review_text,
        "context_variable": context_variable,
        "calculated_score": metric_str
    }

def prepare_freshness_variables(row, f_score_val):
    """Component 2 (Freshness Decay) [NEW]: Logic for Prompt Variables."""
    target_review = str(row.get('text', '')).strip()
    if not target_review or target_review.lower() == 'nan':
        target_review = "[NO TEXT CONTENT]"

    user_name = str(row.get('user/name', 'Unknown User'))
    user_loc = str(row.get('user/userLocation/name', 'Unknown Location'))
    contribs = str(row.get('user/contributions/totalContributions', '0'))
    
    context_variable = (
        f"User: {user_name} | "
        f"Location: {user_loc} | "
        f"Total Contributions: {contribs} | "
        f"Travel Date: {row.get('travelDate', 'N/A')} | "
        f"Publish Date: {row.get('publishedDate', 'N/A')}"
    )

    calculated_score = f"{float(f_score_val):.4f}"

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

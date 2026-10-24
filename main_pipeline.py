import pandas as pd
import asyncio
import json
import os
import sys
import warnings
import re 
import html
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from schemas import MasterFraudAnalysis
from dotenv import load_dotenv

# --- IMPORT YOUR OPTIMIZED LAYERS ---
import Layer1 as L1
import Layer2 as L2
import Layer3 as L3
import Layer4 as L4
import Layer5 as L5

# --- CONFIG ---


load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not API_KEY:
    raise SystemExit("No API key found. Put OPENROUTER_API_KEY=... in a .env file.")

MODEL_NAME = "deepseek/deepseek-v4-flash-0731"
INPUT_FILE = "Final Dataset.xlsx"
OUTPUT_FILE = "Master_Fraud_Results.xlsx"

# 🚀 OPTIMIZATION SETTINGS
CONCURRENCY_LIMIT = 25   # Max speed for 8B model
CHECKPOINT_INTERVAL = 100 # Save CSV every 100 rows (Reduces Disk Lag)

# OpenRouter Specific Headers
YOUR_SITE_URL = "https://fraud-research.com"
YOUR_SITE_NAME = "Fraud Detection Research"

# --- HELPER FUNCTION: PREVENTS EXCEL CRASHES ---
def clean_illegal_chars(val):
    """Removes invisible control characters that crash Excel."""
    if isinstance(val, str):
        # Remove ASCII 0-31 (control chars) except Tab (9) and Newline (10,13)
        return re.sub(r'[\000-\010]|[\013-\014]|[\016-\037]', '', val)
    return val

# Suppress Pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CLIENT SETUP ---
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    default_headers={
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_SITE_NAME,
    }
)

input_lock = asyncio.Lock()

# --- THE MASTER SYSTEM PROMPT (OPTIMIZED & BRIEF) ---
# Updated to prevent "Length Limit" errors on long reviews
MASTER_SYSTEM_PROMPT = f"""
You are the Lead Fraud Detection System. Judge the review using the EVIDENCE provided. No scores are given — form your own judgement from the evidence.

### SECTION 1: BEHAVIORAL & VELOCITY
1. Max Day: {L1.SYSTEM_PROMPT_MAX_DAY}
2. Burstiness: {L1.SYSTEM_PROMPT_BURST}
3. Gap Efficiency: {L1.SYSTEM_PROMPT_FGAP}
4. Burner Account: {L1.SYSTEM_PROMPT_FBURNER}
5. Authority Score: {L1.SYSTEM_PROMPT_FAUTHORITY}

### SECTION 2: LINGUISTIC & STYLE
6. Self-Plagiarism: {L2.SYSTEM_PROMPT_FCOPY}
7. Style Consistency: {L2.SYSTEM_PROMPT_FSTYLE}
8. Smart Caps: {L2.SYSTEM_PROMPT_FCAPS}
9. Lexical Noise: {L2.SYSTEM_PROMPT_FNOISE}
10. Diversity (Richness): {L2.SYSTEM_PROMPT_FRICH}
11. Abstractness: {L2.SYSTEM_PROMPT_FABSTR}
12. Pronoun Usage: {L2.SYSTEM_PROMPT_FPRONOUN}
13. Hype Density: {L2.SYSTEM_PROMPT_FHYPE}

### SECTION 3: SENTIMENT & RATING
14. Positive Ratio: {L3.SYSTEM_PROMPT_FPOS}
15. Negative Ratio: {L3.SYSTEM_PROMPT_FNEG}
16. Extreme Ratio: {L3.SYSTEM_PROMPT_FEXTREME}
17. Deviation: {L3.SYSTEM_PROMPT_FDEV}
18. Sentiment Mismatch: {L3.SYSTEM_PROMPT_FSENTIMENT}

### SECTION 4: TEMPORAL PATTERNS
19. Weighted Bias: {L4.SYSTEM_PROMPT_WEIGHTED_BIAS}
20. One-Hit Extremist: {L4.SYSTEM_PROMPT_ONE_HIT}
21. Rapid Repeat: {L4.SYSTEM_PROMPT_REPEAT_RAPID}
22. Freshness Decay: {L4.SYSTEM_PROMPT_FRESHNESS}

### SECTION 5: SPATIAL & CONSENSUS
23. Location Dist: {L5.SYSTEM_PROMPT_LOC_DIST}
24. Conflict/Bashing: {L5.SYSTEM_PROMPT_CONFLICT}
25. Local Consensus: {L5.SYSTEM_PROMPT_LOCAL_SIM}

### OUTPUT RULES:
- Output strictly valid JSON.
- "is_fake_suspect": "true" if the evidence looks fraudulent, "false" if it looks normal, "insufficient" if the evidence needed for that feature is missing or too thin to judge.
- "consistency_score": your confidence in that verdict, 0.0-1.0. It is NOT a fraud probability. Use a low value when unsure, and 0.0 when the verdict is "insufficient".
- **CRITICAL:** Keep "reasoning" EXTREMELY BRIEF (under 10 words) to save tokens.
- **CRITICAL:** Do NOT exceed 4000 tokens.
"""

HISTORY_DEPTH = 5


def precompute_all_features(df):
    """Runs all Python-side math calculators before the LLM."""
    print("--- ⚡ Phase 1: Pre-computing Math Features ---")
    hotel_column_name = 'placeInfo/name' 
    
    # Layer 1
    df['L1_Authority'] = L1.calculate_social_authority_score(df)
    df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
    date_counts = df['publishedDate'].value_counts()
    df['L1_SameDayCount'] = df['publishedDate'].map(date_counts)
    df['L1_Burst'] = df['L1_SameDayCount'].apply(lambda x: 1 if x > 2 else 0)
    
    # Layer 3
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['hotel_specific_avg'] = df.groupby(hotel_column_name)['rating'].transform('mean')
    df['L3_Dev'] = df.apply(lambda r: L3.calculate_f_dev_score(r, r['hotel_specific_avg']), axis=1)
    
    # Layer 4
    df['L4_WeightDev'] = L4.calculate_weighted_early_bias(df)
    df['L4_OneHit'] = L4.calculate_one_hit_extremist_score(df)
    df['L4_Freshness'] = L4.calculate_freshness_decay(df)
    df['L4_Rapid'] = df.apply(lambda r: L4.calculate_f_repeat_rapid(r, r.get(hotel_column_name, '')), axis=1)

    # Layer 5
    df['L5_Loc'] = L5.precompute_geo_scores(df)
    df['L5_Conflict'] = L5.calculate_geo_conflict_score(
        df, region_column=L5.detect_hotel_region_column(df))

    sim_results = L5.precompute_local_consensus_features(df)
    df['L5_Sim_Score'] = sim_results['Sim_Score']
    df['L5_Sim_Context'] = sim_results['Sim_Context_Str']
    
    return df

async def process_single_review_safe(client, row, sem):
    """
    Robust wrapper with SPLIT RETRY LOGIC:
    1. Connection Errors -> Retry Infinitely (Wait for internet)
    2. Validation/Length Errors -> Retry 5 times, then Skip (Prevent stuck loop)
    """
    async with sem:
        validation_retries = 0
        conn_retries = 0
        MAX_VAL_RETRIES = 5  
        math_payload = {}

        while True:
            try:
                # --- 1. DATA EXTRACTION ---
                history_texts_FULL = []
                history_texts_SHORT = []
                history_ratings = []
                history_dates = []                
                ev_dates, ev_ratings, ev_hotels, ev_locs, ev_texts = [], [], [], [], []

                def safe_str(val):
                    return json.dumps(str(val))[1:-1]

                def _norm(s):
                    """Collapse whitespace and lowercase before comparing.
                    The target 'text' column keeps real newlines, but the
                    Review_i_Text columns have them flattened to spaces, so a
                    raw compare misses 21% of self-inclusions."""
                    return re.sub(r'\s+', ' ', str(s or '')).strip().lower()

                _target_text = _norm(row.get('text', ''))
                _target_date = str(row.get('publishedDate', '') or '').strip()

                for i in range(1, HISTORY_DEPTH + 1):
                    t_col = f'Review_{i}_Text'
                    r_col = f'Review_{i}_Rating'
                    d_col = f'Review_{i}_Date'

                    _h_text = str(row[t_col]).strip() if (t_col in row and pd.notna(row[t_col])) else ''
                    _h_date = str(row[d_col]).strip() if (d_col in row and pd.notna(row[d_col])) else ''

                    # Is this history slot actually the target review itself?
                    is_self = False
                    if _h_text and _target_text and _norm(_h_text)[:120] == _target_text[:120]:
                        is_self = True
                    elif _h_date and _target_date and _h_date == _target_date and not _h_text:
                        is_self = True
                    if is_self:
                        continue

                    truncated = None
                    if _h_text:
                        history_texts_FULL.append(_h_text)
                        clean_text = safe_str(_h_text)
                        truncated = clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
                        history_texts_SHORT.append(truncated)

                    if r_col in row and pd.notna(row[r_col]):
                        try: history_ratings.append(float(row[r_col]))
                        except: pass
                    if _h_date:
                        history_dates.append(_h_date)

                    # aligned evidence lists for the LLM — always append, even None
                    _hotel = row.get(f'Review_{i}_Target_Hotel')
                    _loc = row.get(f'Review_{i}_Location')
                    ev_dates.append(_h_date or None)
                    ev_ratings.append(float(row[r_col]) if (r_col in row and pd.notna(row[r_col])) else None)
                    ev_hotels.append(html.unescape(str(_hotel).strip()) if pd.notna(_hotel) else None)
                    ev_locs.append(str(_loc).strip() if pd.notna(_loc) else None)
                    ev_texts.append(truncated)

                voice_profile_prompt = str(history_texts_SHORT)

                # --- 2. CALCULATORS ---
                l1_vars = L1.prepare_max_day_variables(row) 
                f_gap = L1.calculate_time_gap_efficiency(history_dates)
                f_burner = L1.calculate_burner_score(row)
                
                f_copy = L2.calculate_f_copy(history_texts_FULL, row.get('text', ''))
                f_style = L2.calculate_style_consistency(row.get('text', ''), history_texts_FULL)
                f_caps = L2.calculate_smart_caps_score(row.get('text', ''))
                f_noise = L2.calculate_lexical_noise(row.get('text', ''))
                f_rich = L2.calculate_lexical_diversity(row.get('text', ''))
                            
                f_abstr = L2.calculate_f_abstr(row.get('text', ''))
                f_pronoun = L2.calculate_self_reference_ratio(row.get('text', ''))
                f_hype = L2.calculate_f_hype(row.get('text', ''))

                f_pos = L3.calculate_f_pos(history_ratings)
                f_neg_data = L3.calculate_negative_ratio_score(history_ratings)
                f_extreme = L3.calculate_f_extreme(row)
                
                f_sentiment = L3.calculate_f_sentiment(row)
                
                # --- 3. PACKAGING MATH SCORES ---
                math_payload = {
                    "L1_MaxDay_Math": l1_vars['calculated_score'],
                    "L1_Gap_Math": f_gap,
                    "L1_Burner_Math": f_burner,
                    "L2_Copy_Math": f_copy,
                    "L2_Style_Math": f_style,
                    "L2_Caps_Math": f_caps,
                    "L2_Noise_Math": f_noise,
                    "L2_Rich_Math": f_rich,
                    "L2_Abstr_Math": f_abstr,
                    "L2_Pronoun_Math": f_pronoun,
                    "L2_Hype_Math": f_hype,
                    "L3_Pos_Math": f_pos,
                    "L3_Neg_Math": f_neg_data.get('f_neg_score'),
                    "L3_Extreme_Math": f_extreme,
                    "L3_Sentiment_Math": f_sentiment
                }

                # --- 4. PROMPT CONSTRUCTION ---
                target_text_safe = safe_str(row.get('text', '[No Text]'))

                def evfmt(lst):
                    """Show NOT AVAILABLE when a history list has no real values.

                    A user whose only review is the target itself ends up with a
                    list like [None, None, None, None] after the self-inclusion
                    skip. That is not evidence -- say so plainly instead.
                    """
                    return lst if any(x is not None for x in lst) else "NOT AVAILABLE"

                def fmt(v):
                    """FIX: render missing metrics as an explicit NOT AVAILABLE.

                    Evidence from the previous run: given a sentinel that looked
                    like a real number (0.00), the model fabricated evidence --
                    39.5% of rows marked suspect on a column that was constant
                    zero, with reasons like "Same city as hotel". Given an
                    explicit null it abstained honestly (1.6%, "insufficient
                    data"). Never hand this model a fake number.
                    """
                    if v is None:
                        return "NOT AVAILABLE"
                    try:
                        if pd.isna(v):
                            return "NOT AVAILABLE"
                    except (TypeError, ValueError):
                        pass
                    return v

                user_content = f"""
                TARGET REVIEW: {target_text_safe}
                TITLE: {fmt(row.get('title'))}
                RATING: {fmt(row.get('rating'))}

                === EVIDENCE ===
                Any field marked NOT AVAILABLE means that data is missing for this
                review. For any feature that needs it, answer "insufficient" with
                consistency_score 0.0. Never guess a value.
                The five History lists are aligned: position 1 in each list is the
                same past review.

                [Account]
                TotalContributions: {fmt(row.get('user/contributions/totalContributions'))}
                HelpfulVotes: {fmt(row.get('user/contributions/helpfulVotes'))}
                JoinDate: {fmt(row.get('Join Date'))}

                [History]
                Dates: {evfmt(ev_dates)}
                Ratings: {evfmt(ev_ratings)}
                Hotels: {evfmt(ev_hotels)}
                Locations: {evfmt(ev_locs)}
                Reviews: {evfmt(ev_texts)}

                [This hotel]
                Name: {fmt(row.get('placeInfo/name'))}
                City: {fmt(row.get('placeInfo/addressObj/city'))}, Country: {fmt(row.get('placeInfo/addressObj/country'))}
                HotelAverageRating: {fmt(row.get('hotel_specific_avg'))}
                HotelTotalReviews: {fmt(row.get('placeInfo/numberOfReviews'))}
                SameDayReviews: {fmt(row.get('L1_SameDayCount'))}

                [Trip]
                TravelDate: {fmt(row.get('travelDate'))}
                PublishedDate: {fmt(row.get('publishedDate').date() if pd.notna(row.get('publishedDate')) else None)}
                TimesVisited: {fmt(row.get('Times Visited'))}

                [Reviewer]
                ReviewerHome: {fmt(row.get('user/userLocation/name'))}
                """

                # --- 5. API CALL ---
                completion = await client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=MasterFraudAnalysis,
                    max_tokens=8192,
                    extra_body={
                        "reasoning": {"enabled": False},
                        "provider": {
                            "require_parameters": True,
                            "order": ["CoreWeave", "Baseten", "DeepInfra"],
                            "allow_fallbacks": True,
                            "data_collection": "deny"
                        }
                    }
                )
                
                result = completion.choices[0].message.parsed
                # SUCCESS
                return row.name, result.model_dump(), math_payload

            except Exception as e:
                error_msg = str(e).lower()
                
                # --- CASE A: MODEL ERRORS (Length / Invalid JSON) ---
                if "validation error" in error_msg or "json" in error_msg or "length" in error_msg or "4096" in error_msg:
                    validation_retries += 1
                    print(f"⚠️ Model Error on Row {row.name} (Attempt {validation_retries}/{MAX_VAL_RETRIES}): {e}")
                    
                    if validation_retries >= MAX_VAL_RETRIES:
                        print(f"❌ SKIPPING Row {row.name} after {MAX_VAL_RETRIES} failures.")
                        return row.name, None, math_payload  # Return None for LLM, Keep Math
                    
                    await asyncio.sleep(1) # Short wait
                    
                # --- CASE B: CONNECTION ERRORS (Internet Down) ---
                elif "connect" in error_msg or "timeout" in error_msg or "503" in error_msg:
                    conn_retries += 1
                    if conn_retries > 20:   # ~5 minutes
                        print(f"❌ Connection down for Row {row.name}. Giving up on this row.")
                        return row.name, None, math_payload
                    print(f"⏳ Connection lost on Row {row.name} ({conn_retries}/20). Retrying in 15s...")
                    await asyncio.sleep(15)

                # --- CODE BUGS: retrying can never help ---
                elif isinstance(e, (NameError, AttributeError, TypeError, KeyError, IndentationError)):
                    print(f"❌ CODE BUG on Row {row.name}: {type(e).__name__}: {e}")
                    raise

                # --- AUTH ERRORS: never retryable ---
                elif "401" in error_msg or "403" in error_msg or "api key" in error_msg or "authenticate" in error_msg:
                    print(f"❌ AUTH FAILED — check your API key or credit balance. Stopping.")
                    raise SystemExit(1) 
                   
                # --- CASE C: UNKNOWN ERRORS ---
                else:
                    print(f"⚠️ Generic Error on Row {row.name}: {e}")
                    await asyncio.sleep(2)

async def main():
    # 1. Load Data
    try:
        df = pd.read_excel(INPUT_FILE) if INPUT_FILE.endswith('.xlsx') else pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Input file not found."); return

    # Resume Logic
    start_index = 0
    done_mask = None
    resume_file = None
    if os.path.exists("checkpoint_temp.csv"):
        resume_file = "checkpoint_temp.csv"
    elif os.path.exists(OUTPUT_FILE):
        resume_file = OUTPUT_FILE

    if resume_file:
        existing_df = pd.read_csv(resume_file) if resume_file.endswith('.csv') else pd.read_excel(resume_file)
        if 'f_max_day_suspect' in existing_df.columns:
            done_mask = existing_df['f_max_day_suspect'].notna()
            done_mask.index = existing_df.index
            print(f"--- 🔄 Found {int(done_mask.sum())} completed rows ---")

    # 2. Pre-compute Math
    # Cache the precomputed features. f_local_sim refits a TF-IDF vectorizer
    # per row, which takes ~30 min on the full dataset -- without this, every
    # restart after a dropped connection pays that cost again.
    CACHE_FILE = "precomputed.pkl"
    if os.path.exists(CACHE_FILE):
        print(f"--- 💾 Loading cached features from {CACHE_FILE} ---")
        df = pd.read_pickle(CACHE_FILE)
    else:
        df = precompute_all_features(df)
        df.to_pickle(CACHE_FILE)
        print(f"--- 💾 Cached features to {CACHE_FILE} ---")
    
    # 3. Process
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    if done_mask is not None:
        keep = ~done_mask.reindex(df.index, fill_value=False)
        rows_to_process = df[keep]
    else:
        rows_to_process = df
    
    if rows_to_process.empty:
        print("🎉 All rows already processed!"); return

    print(f"--- ⚡ Phase 2: Processing {len(rows_to_process)} remaining rows ---")
    for index, row in rows_to_process.iterrows():
        row.name = index 
        #calls the function to process each review 
        tasks.append(process_single_review_safe(client, row, sem))



    full_results_df = df.copy() 
    if resume_file and done_mask is not None and done_mask.any():
         full_results_df = pd.read_csv(resume_file) if resume_file.endswith('.csv') else pd.read_excel(resume_file)

    counter = 0
    try:
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            idx, res_dict, math_dict = await future 
            
            # 1. Save Math Scores
            if math_dict:
                for k, v in math_dict.items():
                    full_results_df.at[idx, k] = v
            
            # 2. Save LLM Scores
            if res_dict:
                for feature, data in res_dict.items():
                    if data:
                        _suspect = data.get('is_fake_suspect', 'insufficient')
                        _score = data.get('consistency_score', None)
                        # The model does not reliably zero this on abstentions,
                        # so enforce the rule here instead of trusting the prompt.
                        if _suspect == "insufficient":
                            _score = 0.0
                        full_results_df.at[idx, f"{feature}_score"] = _score
                        full_results_df.at[idx, f"{feature}_suspect"] = _suspect
                        full_results_df.at[idx, f"{feature}_reason"] = data.get('reasoning', "No reasoning")
                    else:
                        # Defaults — this branch means the call failed, so the
                        # score stays blank to distinguish it from an abstention.
                        full_results_df.at[idx, f"{feature}_score"] = None
                        full_results_df.at[idx, f"{feature}_suspect"] = "insufficient"
                        full_results_df.at[idx, f"{feature}_reason"] = "Skipped"
            
            counter += 1
            
            # --- 🚀 CHECKPOINT LOGIC (FAST CSV) ---
            if counter % CHECKPOINT_INTERVAL == 0:
                try:
                    # We save the full dataframe to CSV (Fast)
                    full_results_df.to_csv("checkpoint_temp.csv", index=False)
                    print(f" [Checkpoint] Saved {counter} processed rows to CSV.")
                except Exception as e:
                    print(f" [Warning] Checkpoint save failed: {e}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        raise
    except Exception as e:
        print(f"\n⚠️ Run stopped: {type(e).__name__}: {e}")
        raise
    finally:
        # Runs whether the loop finished, crashed, or was interrupted.
        try:
            full_results_df.to_csv("checkpoint_temp.csv", index=False)
            print(f"💾 Checkpoint saved ({counter} rows this session). Rerun to continue.")
        except Exception as e:
            print(f"⚠️ Could not save checkpoint: {e}")

    # --- 3. FINAL REORDERING & SAVE ---
    print("--- 🧹 Reordering Columns for readability ---")
    
    feature_map = [
        ('f_max_day',       ['L1_MaxDay_Math']),
        ('f_burst',         ['L1_Burst']),
        ('f_gap',           ['L1_Gap_Math']),
        ('f_burner',        ['L1_Burner_Math']),
        ('f_authority',     ['L1_Authority']),
        ('f_copy',          ['L2_Copy_Math']),
        ('f_style',         ['L2_Style_Math']),
        ('f_caps',          ['L2_Caps_Math']),
        ('f_noise',         ['L2_Noise_Math']),
        ('f_rich',          ['L2_Rich_Math']),
        ('f_abstr',         ['L2_Abstr_Math']),
        ('f_pronoun',       ['L2_Pronoun_Math']),
        ('f_hype',          ['L2_Hype_Math']),
        ('f_pos',           ['L3_Pos_Math']),
        ('f_neg',           ['L3_Neg_Math']),
        ('f_extreme',       ['L3_Extreme_Math']),
        ('f_dev',           ['L3_Dev', 'hotel_specific_avg']),
        ('f_sentiment',     ['L3_Sentiment_Math']),
        ('f_weight_dev',    ['L4_WeightDev']),
        ('f_hit_extremist', ['L4_OneHit']),
        ('f_repeat_rapid',  ['L4_Rapid']),
        ('f_fresh',         ['L4_Freshness']),
        ('f_loc_dist',      ['L5_Loc']),
        ('f_conflict',      ['L5_Conflict']),
        ('f_local_sim',     ['L5_Sim_Score', 'L5_Sim_Context'])
    ]
    
    all_feature_cols = []
    for f, math_cols in feature_map:
        all_feature_cols.extend(math_cols)
        all_feature_cols.extend([f"{f}_score", f"{f}_suspect", f"{f}_reason"])
        
    metadata_cols = [c for c in full_results_df.columns if c not in all_feature_cols]
    
    final_order = metadata_cols.copy()
    for f, math_cols in feature_map:
        for mc in math_cols:
            if mc in full_results_df.columns: final_order.append(mc)
        final_order.extend([f"{f}_score", f"{f}_suspect", f"{f}_reason"])
        
    full_results_df = full_results_df.reindex(columns=final_order)

    # --- 🧼 CLEANING & FINAL EXCEL SAVE ---
    print("--- 🧼 Cleaning illegal characters to prevent Excel crash ---")
    full_results_df = full_results_df.map(clean_illegal_chars)
    full_results_df.to_csv("Master_Fraud_Results.csv", index=False)   # always works
    try:
        full_results_df.to_excel(OUTPUT_FILE, index=False)
    except PermissionError:
        print(f"⚠️ {OUTPUT_FILE} is open in Excel — CSV saved instead.")
    print(f"\n✅ Done! Final results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nStopped.")
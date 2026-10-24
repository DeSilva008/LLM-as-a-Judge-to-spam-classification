import pandas as pd
import numpy as np
import requests
import json
import time
import re
from tqdm import tqdm
from datetime import timedelta

# ==========================================
# SYSTEM PROMPTS (OPTIMIZED - 30% REDUCTION)
# ==========================================

# 1. F_MAX_DAY (QUOTA DETECTOR) PROMPT
SYSTEM_PROMPT_MAX_DAY = """MaxDay: farms pay per-review daily quotas; travellers post 1-2/day.
Judge Dates for same-day clustering.
If under 2 dates, or dates show only month and year, answer "insufficient"."""

# 2. F_BURST (VOLUME BURSTINESS) PROMPT
SYSTEM_PROMPT_BURST = """Burst: hotels sometimes pay for a flood of positive reviews in one day to bury a bad one.
Judge whether SameDayReviews is abnormally high for a single hotel.
If SameDayReviews is missing, answer "insufficient"."""

# 3. F_GAP (TIME GAP EFFICIENCY) PROMPT
SYSTEM_PROMPT_FGAP = """Gap: real travellers have months between trips; paid workers review continuously.
Judge the average time between Dates.
Months apart is normal. If dates fall in the same month, the true gap is unknown — answer "insufficient".
If under 2 dates, answer "insufficient"."""

# 4. F_BURNER (HIGH-VELOCITY BURNER) PROMPT
SYSTEM_PROMPT_FBURNER = """Burner: burner accounts post their contracted reviews in a few days then go silent.
Judge the span from earliest to latest Date against the user's total review count.
If dates fall in one month, or under 2 dates, answer "insufficient"."""

# 5. F_AUTHORITY (SOCIAL AUTHORITY SCORE) PROMPT
SYSTEM_PROMPT_FAUTHORITY = """Authority: genuine reviewers earn helpful votes; spammers produce volume that nobody finds useful.
Judge HelpfulVotes against TotalContributions — a low hit rate suggests noise.
A user with 1-2 contributions has had no chance to earn votes — answer "insufficient"."""

# ==========================================
# 1. CALCULATORS (UNCHANGED)
# ==========================================

def calculate_max_day_score(review_dates, physical_limit=10.0):
    """Component (Max Day): Calculates the F_max_day score."""
    if not review_dates or len(review_dates) == 0:
        return 0.0

    try:
        dates = pd.to_datetime(review_dates, errors='coerce').dropna()
        if dates.empty:
            return 0.0
        daily_counts = dates.floor('D').value_counts()
        max_daily_count = daily_counts.max()
        f_score = max_daily_count / physical_limit
        return round(float(f_score), 4)

    except Exception as e:
        print(f"Error calculating max_day score: {e}")
        return 0.0

def calculate_burst_score(df: pd.DataFrame, target_index: int) -> int:
    """Component (F_burst): The Calculator."""
    try:
        target_date = df.loc[target_index, 'publishedDate']
        daily_volume = df[df['publishedDate'] == target_date].shape[0]
        if daily_volume > 2:
            return 1
        else:
            return 0
    except Exception as e:
        print(f"Error calculating burst score for index {target_index}: {e}")
        return 0

def calculate_time_gap_efficiency(date_list):
    """Component (F_gap): The Calculator."""
    valid_dates = [d for d in date_list if pd.notnull(d) and str(d).strip() != '']
    if len(valid_dates) < 2:
        # FIX: was -1.0. The rubric reads "< 1.5 days = HIGH RISK", so the old
        # sentinel scored 41% of the dataset as maximum bot risk purely because
        # the scraper captured fewer than two history dates.
        return np.nan

    try:
        dt_series = pd.Series(pd.to_datetime(valid_dates, format="%B %Y", errors='coerce')).dropna()
        if len(dt_series) < 2:
            return np.nan
        dt_sorted = dt_series.sort_values()
        deltas = dt_sorted.diff().dropna()
        gap_days = deltas / pd.Timedelta(days=1)
        f_gap = gap_days.mean()
        return round(float(f_gap), 2)
    except Exception as e:
        print(f"Error calculating F_gap: {e}")
        return np.nan

def calculate_burner_score(row: pd.Series) -> int:
    """Component (F_burner): The Calculator."""
    dates = []
    
    # Parse Target Date
    if pd.notnull(row.get('publishedDate')):
        dates.append(pd.to_datetime(row['publishedDate'], errors='coerce'))
        
    # Parse History Dates
    for i in range(1, 10): 
        col_name = f'Review_{i}_Date'
        if col_name in row and pd.notnull(row[col_name]):
            dates.append(pd.to_datetime(row[col_name], errors='coerce'))
            
    valid_dates = [d for d in dates if pd.notnull(d)]
    
    C = len(valid_dates) 
    
    if C < 3:
        # FIX: was 0 ("not a burner"). With fewer than 3 dates we cannot tell,
        # and 59% of rows fall here.
        return np.nan
        
    min_date = min(valid_dates)
    max_date = max(valid_dates)
    window_days = (max_date - min_date).days
    
    if window_days < 5:
        return 1
        
    return 0

def calculate_social_authority_score(df: pd.DataFrame) -> pd.Series:
    """
    Component (F_authority): The Calculator.
    """
    print("--- Pre-computing F_authority Scores ---")
    if 'user/contributions/helpfulVotes' not in df.columns:
        df['user/contributions/helpfulVotes'] = 0
    if 'user/contributions/totalContributions' not in df.columns:
        df['user/contributions/totalContributions'] = 0

    # FIX: fillna(0) collapsed "no vote data scraped" into "zero votes", so the
    # rubric flagged 65% of the dataset as spammer/noise on missing metadata.
    helpful_votes = pd.to_numeric(df['user/contributions/helpfulVotes'], errors='coerce')
    total_reviews = pd.to_numeric(df['user/contributions/totalContributions'], errors='coerce')

    numerator = np.log(helpful_votes + 1)
    denominator = np.log(total_reviews.fillna(0) + 5)
    f_authority = numerator / denominator  # NaN wherever helpfulVotes is missing
    
    print("--- F_authority Calculation Complete ---")
    return f_authority

# ==========================================
# 2. PROMPT VARIABLES
# ==========================================

def prepare_max_day_variables(row):
    """Component (Max Day): Logic for Prompt Variables"""
    target_review = str(row.get('text', '')).strip()
    history_logs = []
    all_dates = []
    if 'publishedDate' in row and pd.notna(row['publishedDate']):
        all_dates.append(str(row['publishedDate']))
    for i in range(1, 20): 
        date_col = f'Review_{i}_Date'
        text_col = f'Review_{i}_Text'
        if date_col in row.index and pd.notna(row[date_col]):
            date_val = str(row[date_col])
            raw_text = row.get(text_col)
            text_val = str(raw_text).strip() if pd.notna(raw_text) else "No text"
            # Truncate text to save tokens
            text_val_short = text_val[:100] + "..." if len(text_val) > 100 else text_val
            all_dates.append(date_val)
            history_logs.append(f"[{date_val}] {text_val_short}")
    calculated_score = calculate_max_day_score(all_dates)
    context_variable = "\n".join(history_logs)
    if not context_variable:
        context_variable = "No prior history available."
    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

def prepare_burst_variables(df: pd.DataFrame, target_index: int, f_burst_score: int):
    """Component (F_burst): Prompt Logic."""
    row = df.loc[target_index]
    target_date = row['publishedDate']
    target_review = f"Date: {target_date}\nRating: {row.get('rating', 'N/A')}\nText: {str(row.get('text', ''))}"
    same_day_reviews = df[(df['publishedDate'] == target_date) & (df.index != target_index)]
    
    if not same_day_reviews.empty and f_burst_score == 1:
        history_list = []
        for _, r in same_day_reviews.iterrows():
            r_text = str(r['text']) if pd.notna(r['text']) else "[No Text]"
            snippet = r_text[:100] + "..." if len(r_text) > 100 else r_text
            history_list.append(f"- [Rating: {r.get('rating','N/A')}] {snippet}")
        context_variable = "Concurrent Reviews (Same 24h Window):\n" + "\n".join(history_list)
    elif f_burst_score == 0:
        context_variable = "Volume is organic. No concurrent spike detected."
    else:
        context_variable = "No other reviews found in this window."

    calculated_score_str = f"{f_burst_score} (TRUE)" if f_burst_score == 1 else f"{f_burst_score} (FALSE)"
    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score_str
    }

def prepare_fgap_variables(row):
    """Component (F_gap): Prompt Logic."""
    target_review = f"Review: {row.get('text', '')} | Date: {row.get('publishedDate', '')}"
    history_dates = []
    formatted_history = []
    if 'publishedDate' in row and pd.notnull(row['publishedDate']):
        history_dates.append(row['publishedDate'])
    for i in range(1, 11):
        date_col = f'Review_{i}_Date'
        text_col = f'Review_{i}_Text'
        title_col = f'Review_{i}_Title'
        if date_col in row and pd.notnull(row[date_col]):
            date_val = row[date_col]
            history_dates.append(date_val)
            r_text = str(row.get(text_col, ''))[:50] + "..." if pd.notnull(row.get(text_col)) else "No Text"
            r_title = str(row.get(title_col, '')) if pd.notnull(row.get(title_col)) else "No Title"
            formatted_history.append(f"[{date_val}] {r_title}: {r_text}")

    f_gap_score = calculate_time_gap_efficiency(history_dates)

    if pd.isna(f_gap_score):
        score_str = "Undefined (Insufficient History)"
    else:
        score_str = f"{f_gap_score} Days"

    variables = {
        "target_review": target_review,
        "context_variable": "\n".join(formatted_history) if formatted_history else "No History",
        "calculated_score": f"{f_gap_score} Days" if f_gap_score != -1.0 else "Undefined (Insufficient History)"
    }
    return variables

def prepare_fburner_variables(row):
    """Component (F_burner): Prompt Logic."""
    target_text = row.get('text', '')
    target_date = row.get('publishedDate', 'Unknown Date')
    
    if pd.isna(target_text):
        target_review_str = "[Content Missing/Redacted]"
    else:
        target_review_str = f"Date: {target_date} // Content: {str(target_text).strip()}"

    history_entries = []
    for i in range(1, 6): 
        prefix = f"Review_{i}"
        if f"{prefix}_Text" in row and pd.notnull(row[f"{prefix}_Text"]):
            h_date = row.get(f"{prefix}_Date", "Unknown Date")
            h_loc = row.get(f"{prefix}_Location", "Unknown Loc")
            h_text = str(row[f"{prefix}_Text"]).strip()[:150] 
            entry = f"   - History Item #{i}: [{h_date}] @ {h_loc}: \"{h_text}...\""
            history_entries.append(entry)

    if history_entries:
        context_variable_str = "USER HISTORICAL REVIEWS:\n" + "\n".join(history_entries)
    else:
        context_variable_str = "USER HISTORY: No previous history available."

    calculated_score = calculate_burner_score(row)
    
    return {
        "target_review": target_review_str,
        "context_variable": context_variable_str,
        "calculated_score": calculated_score
    }

def prepare_fauthority_variables(row):
    """Component (F_authority): Prompt Logic."""
    target_review = row.get('text', "")
    if pd.isna(target_review) or target_review == "":
        target_review = "[CONTENT MISSING]"
    
    history_snippets = []
    for i in range(1, 5): 
        col_name = f'Review_{i}_Text'
        if col_name in row and pd.notna(row[col_name]):
            snippet = str(row[col_name])[:300] 
            history_snippets.append(f"History {i}: {snippet}...")
            
    if history_snippets:
        context_variable = "\n".join(history_snippets)
    else:
        context_variable = "No historical review text available."

    score_val = row.get('F_authority_score', 0.0)
    calculated_score = f"{float(score_val):.4f}"

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

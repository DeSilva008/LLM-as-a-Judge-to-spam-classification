import pandas as pd
import numpy as np
import requests
import json
import time
import re
from tqdm import tqdm
from typing import List, Union
# [NEW IMPORT]
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# [NEW INIT] Initialize VADER analyzer once to optimize performance
analyzer = SentimentIntensityAnalyzer()

# ==========================================
# SYSTEM PROMPTS (OPTIMIZED - 40% REDUCTION)
# ==========================================

# 1. F_POS (CHEERLEADER DETECTOR) PROMPT
SYSTEM_PROMPT_FPOS = """Pos: reputation-boosting agents post only 4-5 stars; real travellers occasionally have an average stay.
Judge the spread of Ratings in the user's History.
If History has under 3 ratings, answer "insufficient" — a short history is not evidence of bias."""

# 2. F_NEG (NEGATIVE RATIO) PROMPT
SYSTEM_PROMPT_FNEG = """Neg: agents hired to attack a competitor post only 1-2 star reviews.
Judge the spread of Ratings in the user's History.
If History has under 3 ratings, answer "insufficient" — a short history is not evidence of bias."""

# 3. F_EXTREME (EXTREME RATING RATIO) PROMPT
SYSTEM_PROMPT_FEXTREME = """Extreme: spammers deal in absolutes — all 1s or all 5s — because a 3-star review moves nothing.
Judge whether the user's History Ratings show any middle ground, or only extremes.
If History has under 3 ratings, answer "insufficient"."""

# 4. F_DEV (GENERAL RATING DEVIATION) PROMPT
SYSTEM_PROMPT_FDEV = """Dev: a rating far from the hotel's consensus average can signal an agenda.
Judge the target Rating against the hotel's average rating.
If either is missing, answer "insufficient" — and note that a lone dissenting rating can also be an honest bad experience."""

# 5. F_SENTIMENT (SENTIMENT POLARITY) PROMPT
SYSTEM_PROMPT_FSENTIMENT = """Sentiment: a mismatch between the text's emotional tone and the star rating suggests a bot or careless paste.
Judge whether the target review's tone matches its Rating.
If the review is under 20 words, answer "insufficient"."""

# ==========================================
# 1. CALCULATORS (UNCHANGED)
# ==========================================

def calculate_f_pos(ratings: Union[List[float], np.ndarray]) -> float:
    """ Component 1 (F_Pos): Calculates the Calibrated Positive Ratio. """
    clean_ratings = np.array([r for r in ratings if pd.notna(r)], dtype=float)
    N = len(clean_ratings)
    # FIX: gate was 20, but only 5 history columns exist -- this returned 0.0 for
    # 100% of rows. 0.0 also means "0% positive", i.e. the hater extreme.
    if N < 3:
        return np.nan
    positive_count = np.sum(clean_ratings >= 4)
    f_pos_score = positive_count / N
    return float(f_pos_score)

def calculate_negative_ratio_score(ratings_list):
    """ Component 1 (F_Neg): Calculates the Negative Ratio score. """
    clean_ratings = [r for r in ratings_list if pd.notna(r) and isinstance(r, (int, float))]
    
    total_count = len(clean_ratings)
    if total_count == 0:
        return {"f_neg_score": np.nan, "total_count": 0, "risk_label": "Insufficient Data"}

    negative_reviews = sum(1 for r in clean_ratings if r <= 2)
    f_neg = negative_reviews / total_count
    
    label = "Safe"
    if f_neg > 0.65:
        if total_count < 3:
            label = "Burner Risk"
        else:
            label = "The Hater"
    elif f_neg <= 0.40:
        label = "Safe"
    else:
        label = "Moderate Concern"

    return {"f_neg_score": round(f_neg, 3), "total_count": total_count, "risk_label": label}

def calculate_f_extreme(row):
    """ Component 1 (F_Extreme): Calculates extreme rating ratio. """
    all_ratings = []
    try:
        current_rating = float(row.get('rating', np.nan))
        if not pd.isna(current_rating): all_ratings.append(current_rating)
    except (ValueError, TypeError): pass

    history_pattern = re.compile(r'^Review_\d+_Rating$')
    history_cols = [c for c in row.index if history_pattern.match(str(c))]
    
    for col in history_cols:
        try:
            val = float(row[col])
            if not pd.isna(val): all_ratings.append(val)
        except (ValueError, TypeError): continue

    total_reviews = len(all_ratings)
    # FIX: gate was 5 (63.8% of rows fell through it), sentinel was 0.0 which is
    # also a legitimate value meaning "no extreme ratings".
    if total_reviews < 3: return np.nan
    
    extreme_count = sum(1 for r in all_ratings if r == 1.0 or r == 5.0)
    return float(extreme_count / total_reviews)

def calculate_f_dev_score(row, hotel_avg):
    """ Component 1 (F_Dev): The Calculator. """
    try:
        user_rating = float(row.get('rating', np.nan))
        if np.isnan(user_rating):
            return 0.0
        dev_score = abs(user_rating - hotel_avg) / 4.0
        return round(dev_score, 4)
    except (ValueError, TypeError):
        return 0.0

def calculate_f_sentiment(row):
    """Component 1 (F_sentiment): Calculates the F_sentiment score."""
    try:
        star_rating = float(row.get('rating', 0))
        review_text = str(row.get('text', ""))
    except (ValueError, TypeError):
        return np.nan

    if not review_text or review_text.lower() == 'nan':
        return np.nan

    vs = analyzer.polarity_scores(review_text)
    vader_score = vs['compound']

    # FIX: the old binary fired on 1.0% of rows. `stars >= 4.5` only ever meant
    # 5-star (ratings are integers), and VADER rarely reaches -0.5 on long hotel
    # prose, so the conjunction was near-empty. This is the continuous mismatch:
    # 0.0 = text and stars agree, 1.0 = maximum contradiction.
    expected = (star_rating - 3.0) / 2.0      # maps 1..5 stars onto -1..+1
    return round(abs(vader_score - expected) / 2.0, 4)

# ==========================================
# 2. PROMPT VARIABLES (UNCHANGED)
# ==========================================

def prepare_fpos_variables(row):
    """ Component 2 (F_Pos): Logic for Prompt Variables """
    target_text = str(row.get('text', '')) if pd.notna(row.get('text')) else "[No Text Content]"
    target_rating = row.get('rating', 0)
    
    history_ratings = []
    history_texts = []
    
    for i in range(1, 101):
        rating_col = f'Review_{i}_Rating'
        text_col = f'Review_{i}_Text'
        if rating_col in row and pd.notna(row[rating_col]):
            try:
                r_val = float(row[rating_col])
                history_ratings.append(r_val)
                t_val = str(row.get(text_col, "")).strip()
                if t_val: history_texts.append(f"[Review {i} | {int(r_val)} Stars]: {t_val}")
            except ValueError: continue
        else:
            if i > 5 and len(history_ratings) == 0: break 

    calculated_score = calculate_f_pos(history_ratings)
    context_variable = "\n".join(history_texts) if history_texts else "No historical text available."

    return {
        "target_review": f"[Current | {target_rating} Stars]: {target_text}",
        "context_variable": context_variable,
        "calculated_score": round(calculated_score, 4)
    }

def prepare_fneg_variables(row):
    """ Component 2 (F_Neg): Logic for Prompt Variables """
    target_review_text = str(row.get('text', ''))
    target_review_rating = row.get('rating', 0)
    
    history_ratings = []
    history_texts = []
    
    for i in range(1, 101): 
        rating_col = f'Review_{i}_Rating'
        text_col = f'Review_{i}_Text'
        if rating_col in row and pd.notna(row[rating_col]):
            try:
                r_val = float(row[rating_col])
                history_ratings.append(r_val)
                rev_txt = str(row.get(text_col, "No text")).strip()
                rev_txt = (rev_txt[:150] + '..') if len(rev_txt) > 150 else rev_txt
                history_texts.append(f"[Rating: {int(r_val)}]: {rev_txt}")
            except ValueError: continue
        else:
            if i > 5 and len(history_ratings) == 0: break

    full_history_ratings = history_ratings + [target_review_rating]
    calculation_result = calculate_negative_ratio_score(full_history_ratings)
    voice_profile = "\n".join(history_texts) if history_texts else "No prior textual history available."

    return {
        "target_review": target_review_text,
        "context_variable": voice_profile,
        "calculated_score": f"{calculation_result['f_neg_score']} (Count: {calculation_result['total_count']}, Label: {calculation_result['risk_label']})",
        "_raw_score": calculation_result['f_neg_score'],
        "_risk_label": calculation_result['risk_label']
    }

def prepare_fextreme_variables(row):
    """ Component 2 (F_Extreme): Logic for Prompt Variables """
    target_title = str(row.get('title', ''))
    target_text = str(row.get('text', ''))
    target_rating = str(row.get('rating', 'N/A'))
    target_review_str = f"Rating: {target_rating}/5\nTitle: {target_title}\nReview: {target_text}"

    history_entries = []
    i = 1
    while f'Review_{i}_Rating' in row:
        r_rating = row.get(f'Review_{i}_Rating')
        r_title = row.get(f'Review_{i}_Title')
        r_text = row.get(f'Review_{i}_Text')
        if pd.notna(r_rating):
            text_preview = str(r_text)[:100] + "..." if len(str(r_text)) > 100 else str(r_text)
            history_entries.append(f"Review {i}: [{int(r_rating)}/5] '{str(r_title)}' - {text_preview}")
        i += 1

    context_str = "\n".join(history_entries) if history_entries else "No historical reviews available."
    calc_score = calculate_f_extreme(row)

    return {
        "target_review": target_review_str,
        "context_variable": context_str,
        "calculated_score": f"{calc_score:.2f}"
    }

def prepare_fdev_variables(row, hotel_avg_rating):
    """ Component 2 (F_Dev): Logic for Prompt Variables """
    title = str(row.get('title', '')) if not pd.isna(row.get('title')) else ""
    body = str(row.get('text', '')) if not pd.isna(row.get('text')) else ""
    target_review = f"Title: {title}\nReview: {body}".strip()
    
    f_dev_val = calculate_f_dev_score(row, hotel_avg_rating)
    calculated_score = f"{f_dev_val:.4f}"
    
    user_rating = row.get('rating', 'N/A')
    
    try:
        u_rat_float = float(user_rating)
        diff = abs(u_rat_float - hotel_avg_rating)
        diff_str = f"{diff:.1f}"
    except:
        diff_str = "N/A"

    context_variable = (
        f"Global Hotel Average: {hotel_avg_rating:.2f}/5.0.\n"
        f"User Rating: {user_rating}/5.0.\n"
        f"Deviation: {diff_str} points."
    )

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

def prepare_fsentiment_variables(row):
    """ Component 2 (F_sentiment): Logic for Prompt Variables. """
    title = str(row.get('title', '')).strip()
    body = str(row.get('text', '')).strip()
    
    if title and title.lower() != 'nan':
        target_review = f"Title: {title}\nBody: {body}"
    else:
        target_review = body

    history_list = []
    for i in range(1, 6):
        col_name = f'Review_{i}_Text'
        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != '':
            hist_text = str(row[col_name])[:300] 
            history_list.append(f"[History {i}]: {hist_text}...")

    if history_list:
        context_variable = "\n".join(history_list)
    else:
        context_variable = "No historical reviews available for this user."

    calculated_score = calculate_f_sentiment(row)

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

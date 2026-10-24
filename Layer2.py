import pandas as pd
import numpy as np
import requests
import json
import time
import re
from tqdm import tqdm
import emoji 

# Machine Learning & NLP Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- IMPORTS FOR FEATURE F_ABSTR ---
import spacy
import nltk
from nltk.corpus import wordnet as wn

# Ensure necessary NLTK/Spacy data is available
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading Spacy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
 

# ==========================================
# SYSTEM PROMPTS (OPTIMIZED - 35% REDUCTION)
# ==========================================

# 8. F_COPY (SELF-PLAGIARISM) PROMPT
SYSTEM_PROMPT_FCOPY = """Copy: lazy spammers write one generic review and paste it across many hotels.
Compare the user's own reviews in History for near-identical wording.
If History has under 2 reviews, answer "insufficient"."""

# 10. F_STYLE (AUTHORSHIP CONSISTENCY) PROMPT
SYSTEM_PROMPT_FSTYLE = """Style: hijacked accounts show a sudden change in writing style from the previous owner.
Compare the target review's writing style to the user's History — vocabulary, sentence length, tone.
If History has under 2 reviews, or the reviews are under 40 words, answer "insufficient"."""

# 11. F_CAPS (SMART CAPITALIZATION) PROMPT
SYSTEM_PROMPT_FCAPS = """Caps: shouting in the middle of sentences signals emotional dysregulation or amateur marketing.
Judge the target review for ALL-CAPS words used for emphasis, ignoring proper nouns and sentence starts.
If the review is under 20 words, answer "insufficient"."""

# 12. F_NOISE (LEXICAL NOISE) PROMPT
SYSTEM_PROMPT_FNOISE = """Noise: clickbait titles use shouting, repeated punctuation, and emoji.
Judge the review Title for excessive caps, "!!!" runs, or emoji — normal Title Case is not noise.
If the Title is missing, answer "insufficient"."""

# 13. F_RICH (LEXICAL DIVERSITY) PROMPT
SYSTEM_PROMPT_FRICH = """Rich: content-farm writers reuse a small set of safe words — good, nice, best, clean.
Judge whether the target review's vocabulary is narrow and repetitive rather than specific.
If the review is under 20 words, answer "insufficient"."""

# 14. F_ABSTR (ABSTRACTNESS) PROMPT
SYSTEM_PROMPT_FABSTR = """Abstr: people who never stayed can't name physical specifics — they write abstract praise like "world-class hospitality" instead of "the omelet was cold".
Judge whether the target review contains concrete, checkable detail about the stay.
If the review is under 20 words, answer "insufficient"."""

# 15. F_PRONOUN (SELF-REFERENCE RATIO) PROMPT
SYSTEM_PROMPT_FPRONOUN = """Pronoun: a reviewer describing their own stay usually says "I", "we", "my"; impersonal reviews read like a brochure.
Judge whether the target review is written from personal experience or detached description.
If the review is under 20 words, answer "insufficient"."""

# 16. F_HYPE (HYPERBOLE DENSITY) PROMPT
SYSTEM_PROMPT_FHYPE = """Hype: paid shills write marketing copy — stacked intensifiers and fluff adjectives like "absolutely magical", "truly stunning".
Judge whether the target review reads like promotional copy rather than a guest describing a stay.
If the review is under 20 words, answer "insufficient"."""

# ==========================================
# 1. CALCULATORS (UNCHANGED)
# ==========================================

def calculate_f_copy(history_reviews: list, target_text=None) -> float:
    """Component (F_copy): The Calculator.

    FIX: the target review is now included in the corpus. Previously this
    compared history items only to each other, so a user with a diverse
    history but a copy-pasted CURRENT review scored 0.0 -- and the current
    review is the thing being classified.
    FIX: returns np.nan (not 0.0) when there is nothing to compare against.
    """
    clean_history = [
        str(r).strip() for r in history_reviews 
        if pd.notna(r) and isinstance(r, (str, bytes)) and len(str(r).strip()) > 10
    ]
    has_target = isinstance(target_text, str) and len(target_text.strip()) > 10

    if has_target:
        if len(clean_history) < 1:
            return np.nan
        corpus = [target_text.strip()] + clean_history
    else:
        if len(clean_history) < 2:
            return np.nan
        corpus = clean_history

    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        if has_target:
            # max similarity between the TARGET and any history item
            sims = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
            max_similarity = float(sims.max())
        else:
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0.0)
            max_similarity = float(sim_matrix.max())
        return round(max_similarity, 4)
    except ValueError:
        return np.nan

def calculate_style_consistency(target_text, history_texts):
    """Component (F_style): The Calculator."""
    if not isinstance(target_text, str) or len(target_text.split()) <= 20:
        return np.nan

    valid_history = [t for t in history_texts if isinstance(t, str) and len(t.strip()) > 0]
    if len(valid_history) < 3:
        return np.nan

    all_corpus = [target_text] + valid_history
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_corpus)
        v_r = tfidf_matrix[0]
        history_vectors = tfidf_matrix[1:]
        # FIX: np.mean on a sparse matrix returns np.matrix, which current
        # scikit-learn rejects with TypeError. The bare except below was
        # swallowing it, so this function returned None on EVERY row.
        mu_user = np.asarray(np.mean(history_vectors, axis=0))
        score = cosine_similarity(v_r, mu_user)[0][0]
        return float(round(score, 4))
    except Exception:
        return np.nan

def calculate_smart_caps_score(text):
    """Component (F_caps): The Calculator."""
    if pd.isna(text) or text == "":
        return 0.0

    str_text = str(text)
    total_chars = len(str_text)
    
    if total_chars == 0:
        return 0.0

    count_caps = sum(1 for char in str_text if char.isupper())
    sentences = re.split(r'[.!?]+(?:\s+|$)', str_text)
    count_sentences = len([s for s in sentences if s.strip()])

    numerator = max(0, count_caps - count_sentences)
    f_caps = numerator / total_chars

    return round(f_caps, 4)

def calculate_lexical_noise(text):
    """Component (F_noise): The Calculator."""
    if pd.isna(text) or text == "" or not isinstance(text, str):
        return 0.0

    n_length = len(text)
    n_caps = sum(1 for char in text if char.isupper())
    n_punct = len(re.findall(r'[^\w\s]', text))
    n_emoji = emoji.emoji_count(text)
    
    if n_length == 0:
        return 0.0
        
    raw_score = (n_caps + (2 * n_punct) + (3 * n_emoji)) / n_length
    return round(raw_score, 4)

def calculate_lexical_diversity(text: str) -> float:
    """Component (F_rich): The Calculator"""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = clean_text.split()
    total_words = len(tokens)
    
    if total_words == 0:
        return 0.0
        
    unique_words = len(set(tokens))
    ttr_score = unique_words / total_words
    return round(ttr_score, 4)

def is_abstract_noun(word_token):
    """Helper for F_abstr"""
    synsets = wn.synsets(word_token, pos=wn.NOUN)
    if not synsets:
        abstract_suffixes = ('ness', 'ity', 'tion', 'ment', 'ism', 'ance', 'ence')
        return word_token.lower().endswith(abstract_suffixes)

    synset = synsets[0]
    for path in synset.hypernym_paths():
        names = [s.name().split('.')[0] for s in path]
        if 'abstraction' in names:
            return True
        if 'physical_entity' in names:
            return False
    return False

def calculate_f_abstr(text):
    """Component (F_abstr): The Calculator"""
    if not isinstance(text, str) or len(text.split()) <= 25:
        return np.nan  # FIX: was 0.0, which reads as "fully concrete" = genuine

    doc = nlp(text)
    abstract_count = 0
    concrete_count = 0
    
    for token in doc:
        if token.pos_ == "NOUN":
            if is_abstract_noun(token.text):
                abstract_count += 1
            else:
                concrete_count += 1
    
    total_nouns = abstract_count + concrete_count
    if total_nouns == 0:
        return 0.0
        
    return round(abstract_count / total_nouns, 4)

def calculate_self_reference_ratio(text: str) -> float:
    """Component (F_pronoun): The Calculator."""
    if pd.isna(text) or not isinstance(text, str):
        return 0.0
    
    clean_text = text.lower()
    tokens = re.findall(r'\b\w+\b', clean_text)
    total_words = len(tokens)
    
    if total_words == 0:
        return 0.0
    
    target_pronouns = {"i", "me", "my", "we", "us", "our"}
    pronoun_count = sum(1 for token in tokens if token in target_pronouns)
    
    return pronoun_count / total_words

# --- CALCULATOR COMPONENT FOR F_HYPE ---

HYPE_INTENSIFIERS = {
    "absolutely", "completely", "extremely", "highly", "incredibly", 
    "really", "so", "totally", "truly", "very", "undoubtedly"
}
HYPE_ADJECTIVES = {
    "amazing", "awesome", "breathtaking", "brilliant", "excellent", 
    "fabulous", "fantastic", "gorgeous", "incredible", "lovely", 
    "magical", "magnificent", "marvelous", "outstanding", "perfect", 
    "phenomenal", "spectacular", "stunning", "wonderful", "world-class"
}
ALL_HYPE_TERMS = HYPE_INTENSIFIERS | HYPE_ADJECTIVES
HYPE_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(term) for term in ALL_HYPE_TERMS) + r')\b', re.IGNORECASE)

def calculate_f_hype(text: str) -> float:
    """Component (F_hype): The Calculator."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    words = re.findall(r'\w+', text)
    total_words = len(words)
    
    if total_words < 35:
        return np.nan  # FIX: was 0.0, indistinguishable from "no hype at all"

    match_count = len(HYPE_PATTERN.findall(text))
    score = match_count / total_words
    return round(score, 4)


# ==========================================
# 2. PROMPT VARIABLES (UNCHANGED)
# ==========================================

def prepare_fcopy_variables(row):
    """Component (F_copy): Prompt Logic."""
    target_review = str(row.get('text', '')).strip()
    if pd.isna(row.get('text')):
        target_review = "N/A"

    history_texts = []
    for i in range(1, 10): 
        col_name = f'Review_{i}_Text'
        if col_name in row.index:
            val = row[col_name]
            if pd.notna(val) and len(str(val).strip()) > 0:
                history_texts.append(str(val).strip())

    f_copy_score = calculate_f_copy(history_texts)

    if not history_texts:
        context_variable = "No historical data available."
    else:
        formatted_history = [f"[{i+1}] {text}" for i, text in enumerate(history_texts)]
        context_variable = "\n".join(formatted_history)

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": f_copy_score
    }

def prepare_fstyle_variables(row):
    """Component (F_style): Prompt Logic."""
    target_review = str(row.get('text', '')).strip()
    if pd.isna(target_review) or target_review == "":
        target_review = "N/A"

    history_list = []
    for i in range(1, 6): 
        col = f'Review_{i}_Text'
        val = row.get(col)
        if pd.notna(val) and isinstance(val, str) and val.strip() != "":
            history_list.append(val)

    raw_score = calculate_style_consistency(target_review, history_list)
    
    if raw_score is None:
        calculated_score_str = "NULL (Insufficient Data - Neutral Risk)"
    else:
        calculated_score_str = f"{raw_score:.4f}"

    if not history_list:
        context_variable = "No historical reviews available."
    else:
        formatted_history = [f"[History Sample {i+1}]: {text}" for i, text in enumerate(history_list)]
        context_variable = "\n".join(formatted_history)

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score_str
    }

def prepare_fcaps_variables(row, calculated_score):
    """Component (F_caps): Prompt Logic."""
    target_review = str(row['text']) if pd.notna(row['text']) else "No text content."
    history_reviews = []
    for i in range(2, 6): 
        col_name = f'Review_{i}_Text'
        if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != "":
            snippet = str(row[col_name])[:300] 
            history_reviews.append(f"[History {i-1}]: {snippet}...")

    if history_reviews:
        context_variable = "\n".join(history_reviews)
    else:
        context_variable = "No previous history available for this user."

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

def prepare_fnoise_variables(row, calculated_score):
    """Component (F_noise): Prompt Logic."""
    title = str(row.get('title', '')) if pd.notna(row.get('title')) else ""
    body = str(row.get('text', '')) if pd.notna(row.get('text')) else ""
    target_review_str = f"TITLE: {title}\nBODY: {body}" if title else body

    history_entries = []
    for i in range(2, 5): 
        h_title = row.get(f'Review_{i}_Title')
        h_text = row.get(f'Review_{i}_Text')
        if pd.notna(h_text):
            title_part = f"{h_title} - " if pd.notna(h_title) else ""
            history_entries.append(f"[Historical Review {i-1}]: {title_part}{str(h_text)[:100]}...")

    if history_entries:
        context_str = "\n".join(history_entries)
    else:
        context_str = "No historical reviews available for this user."

    return {
        "target_review": target_review_str,
        "context_variable": context_str,
        "calculated_score": calculated_score
    }

def prepare_frich_variables(row, calculated_score):
    """Component (F_rich): Prompt Logic"""
    target_review = row.get('text', "")
    if pd.isna(target_review):
        target_review = "[NO TEXT AVAILABLE]"

    history_columns = [col for col in row.index if 'Review_' in col and '_Text' in col]
    history_texts = []
    
    for col in history_columns:
        val = row[col]
        if isinstance(val, str) and len(val) > 5:
            clean_val = val.replace('\n', ' ').strip()
            history_texts.append(f"- {clean_val}")

    if history_texts:
        context_variable = "PRIOR USER REVIEWS:\n" + "\n".join(history_texts[:3])
    else:
        context_variable = "PRIOR USER REVIEWS: None available (Cold Start User)."

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": calculated_score
    }

def prepare_fabstr_variables(row, calculated_score):
    """Component (F_abstr): Prompt Logic."""
    target_review = str(row.get('text', ''))
    
    history_columns = ['Review_1_Text', 'Review_2_Text', 'Review_3_Text', 'Review_4_Text']
    history_list = []
    for col in history_columns:
        if col in row.index:
            val = row.get(col)
            if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 0:
                history_list.append(f"- {val.strip()}")
            
    if history_list:
        context_variable = "User Previous Reviews:\n" + "\n".join(history_list)
    else:
        context_variable = "User Previous Reviews: [No History Available]"

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": f"{calculated_score:.2f}"
    }

def prepare_fpronoun_variables(row, calculated_score):
    """Component (F_pronoun): Prompt Logic."""
    target_review = row.get('text', '')
    if pd.isna(target_review):
        target_review = "[NO TEXT CONTENT]"

    history_entries = []
    for i in range(1, 11):
        col_name = f'Review_{i}_Text'
        date_col = f'Review_{i}_Date'
        
        if col_name in row and pd.notna(row[col_name]):
            rev_text = row[col_name]
            rev_date = row.get(date_col, 'Unknown Date')
            
            if len(str(rev_text)) > 300:
                rev_text = str(rev_text)[:300] + "..."
                
            entry = f"- [Date: {rev_date}]: \"{rev_text}\""
            history_entries.append(entry)
            
    if history_entries:
        context_variable = "User Historical Reviews:\n" + "\n".join(history_entries)
    else:
        context_variable = "No historical reviews available for this user."

    return {
        "target_review": target_review,
        "context_variable": context_variable,
        "calculated_score": round(calculated_score, 4) 
    }

def prepare_fhype_variables(row, calculated_score):
    """Component (F_hype): Prompt Logic."""
    target_review = str(row.get('text', '')).strip()
    if target_review.lower() in ['nan', 'none', '']:
        target_review = "[NO TEXT AVAILABLE]"

    history_reviews = []
    for i in range(1, 5):
        col_name = f'Review_{i}_Text'
        if col_name in row and pd.notna(row[col_name]):
            review_text = str(row[col_name]).strip()
            if len(review_text) > 10: 
                history_reviews.append(f"- History Item {i}: \"{review_text[:200]}...\"")

    if history_reviews:
        context_variable = "PRIOR USER REVIEWS:\n" + "\n".join(history_reviews)
    else:
        context_variable = "PRIOR USER REVIEWS: None available (First time reviewer or data missing)."

    return {
        "target_review": target_review,
        "calculated_score": f"{calculated_score:.2f}",
        "context_variable": context_variable
    }


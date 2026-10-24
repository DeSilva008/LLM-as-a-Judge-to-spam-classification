# LLM-as-a-Judge: Multi-Layer Fake Review Detection

## Overview

This project implements a multi-layer framework for identifying potentially fake or suspicious online reviews. It combines mathematically calculated features with Large Language Model (LLM) analysis to evaluate review authenticity from multiple perspectives.

The framework analyzes **25 features across five dimensions**: behavioral, linguistic, sentiment, temporal, and spatial. Each feature uses calculated scores, review content, reviewer history, or contextual information to support the LLM's assessment.

## Key Features

- **Multi-layer analysis:** Evaluates suspicious review patterns across five dimensions.
- **Mathematical feature calculation:** Uses Python-based calculations to generate quantitative indicators.
- **LLM-based evaluation:** Uses system prompts to analyze calculated scores and review context.
- **Reviewer history analysis:** Considers historical reviews, ratings, publication dates, and reviewer contributions.
- **Missing-data handling:** Uses insufficient-data states where evidence is not adequate for reliable assessment.
- **Structured output:** Defines Pydantic models for consistent feature-level and overall analysis results.

## Five Analysis Dimensions

### 1. Behavioral Analysis

Examines reviewer activity patterns and account behavior.

| Feature | Description |
|---|---|
| `f_max_day` | Detects unusually high daily review activity. |
| `f_burst` | Identifies potential bursts of reviews within a short period. |
| `f_gap` | Examines time gaps between reviews. |
| `f_burner` | Looks for accounts posting reviews within a short time window. |
| `f_authority` | Evaluates helpful votes relative to reviewer contributions. |

### 2. Linguistic Analysis

Examines writing characteristics that may indicate suspicious or repetitive content.

The defined features are `f_copy`, `f_style`, `f_caps`, `f_noise`, `f_rich`, `f_abstr`, `f_pronoun`, and `f_hype`.

These features represent checks for copying, writing style, capitalization, textual noise, richness, abstraction, pronoun usage, and promotional language.

### 3. Sentiment Analysis

Examines rating distributions and the relationship between review text and sentiment.

| Feature | Description |
|---|---|
| `f_pos` | Measures the proportion of positive historical ratings. |
| `f_neg` | Measures the proportion of negative ratings. |
| `f_extreme` | Examines the frequency of extreme ratings. |
| `f_dev` | Measures deviation from the hotel's average rating. |
| `f_sentiment` | Measures the mismatch between textual sentiment and the star rating. |

The provided implementation uses VADER sentiment analysis for the sentiment mismatch calculation.

### 4. Temporal Analysis

Examines timing-related patterns that may indicate suspicious review activity.

| Feature | Description |
|---|---|
| `f_weight_dev` | Evaluates rating deviation with weighting based on review order. |
| `f_hit_extremist` | Identifies single-contribution accounts paired with extreme ratings. |
| `f_repeat_rapid` | Detects repeated reviews of the same hotel within a short period. |
| `f_fresh` | Measures the time difference between travel and review publication dates. |

### 5. Spatial Analysis

Examines geographical relationships and similarity to the hotel's review context.

| Feature | Description |
|---|---|
| `f_loc_dist` | Evaluates the relationship between reviewer and hotel locations. |
| `f_conflict` | Looks for potentially suspicious rating patterns involving nearby competing hotels. |
| `f_local_sim` | Measures the semantic similarity between a review and the hotel's recent review context. |

The provided spatial implementation uses TF-IDF vectorization and cosine similarity for local semantic consensus.

## Technologies Used

- **Python** — Core implementation
- **Pandas and NumPy** — Data processing and numerical calculations
- **Requests** — HTTP requests
- **Pydantic** — Structured data validation
- **VADER Sentiment** — Sentiment analysis
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **tqdm** — Progress tracking
- **OpenRouter** — LLM API integration in the main pipeline

## Project Structure

The repository contains the main pipeline, feature-layer modules, and schema definitions.

```text
LLM-as-a-Judge/
├── main_pipeline.py
├── Layer1.py
├── Layer2.py
├── Layer3.py
├── Layer4.py
├── Layer5.py
├── schemas.py
├── README.md
└── .gitignore
```

*This structure reflects the intended organization described for the project. Adjust filenames if your actual repository uses different names.*

## Structured Output

The `schemas.py` module defines two Pydantic models.

### `FeatureResult`

Represents the result for an individual feature:

- `consistency_score`: An optional numerical score.
- `reasoning`: An explanation of the assessment.
- `is_fake_suspect`: A categorical result with three possible values: `true`, `false`, or `insufficient`.

### `MasterFraudAnalysis`

Defines the complete analysis structure, containing fields for all 25 features across the five dimensions.

The schema supports structured feature-level results and an explicit `insufficient` state when the available evidence does not support a confident assessment.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd LLM-as-a-Judge
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages used by the provided modules:

```bash
pip install pandas numpy requests tqdm pydantic vaderSentiment scikit-learn python-dateutil
```

### 4. Configure API access

If your main pipeline uses OpenRouter, configure your API key according to the environment-variable or configuration method used in `main_pipeline.py`.

**Security:** Do not commit API keys, passwords, private datasets, or other secrets to GitHub.

## Running the Pipeline

After installing the dependencies and configuring the required API access, run the main pipeline using:

```bash
python main_pipeline.py
```

The exact input-data format, command-line arguments, output files, and configuration requirements depend on the implementation in `main_pipeline.py`.

## Important Considerations

- A suspicious feature score is an indicator, not definitive proof that a review is fake.
- Missing reviewer history or incomplete metadata can limit the reliability of individual feature assessments.
- LLM-generated assessments depend on the supplied context and prompts.
- The framework should be evaluated against appropriately labeled data before being used for consequential decisions.

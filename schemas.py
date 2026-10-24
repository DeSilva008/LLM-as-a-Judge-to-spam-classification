from pydantic import BaseModel
from typing import Optional
from typing import Literal

# --- ROBUST SCHEMA (Accepts Nulls) ---
class FeatureResult(BaseModel):
    # If LLM returns null, it defaults to 0.0 / False
    consistency_score: Optional[float] = None
    reasoning: Optional[str] = "Insufficient data for analysis"
    is_fake_suspect: Literal["true", "false", "insufficient"] = "insufficient"

# The Master Schema containing ALL 25 layers
class MasterFraudAnalysis(BaseModel):
    # --- DIMENSION 1: BEHAVIORAL ---
    f_max_day: Optional[FeatureResult] = FeatureResult()
    f_burst: Optional[FeatureResult] = FeatureResult()
    f_gap: Optional[FeatureResult] = FeatureResult()
    f_burner: Optional[FeatureResult] = FeatureResult()
    f_authority: Optional[FeatureResult] = FeatureResult()
    
    # --- DIMENSION 2: LINGUISTIC ---
    f_copy: Optional[FeatureResult] = FeatureResult()
    f_style: Optional[FeatureResult] = FeatureResult()
    f_caps: Optional[FeatureResult] = FeatureResult()
    f_noise: Optional[FeatureResult] = FeatureResult()
    f_rich: Optional[FeatureResult] = FeatureResult()
    f_abstr: Optional[FeatureResult] = FeatureResult()
    f_pronoun: Optional[FeatureResult] = FeatureResult()
    f_hype: Optional[FeatureResult] = FeatureResult()
    
    # --- DIMENSION 3: SENTIMENT ---
    f_pos: Optional[FeatureResult] = FeatureResult()
    f_neg: Optional[FeatureResult] = FeatureResult()
    f_extreme: Optional[FeatureResult] = FeatureResult()
    f_dev: Optional[FeatureResult] = FeatureResult()
    f_sentiment: Optional[FeatureResult] = FeatureResult()
    
    # --- DIMENSION 4: TEMPORAL ---
    f_weight_dev: Optional[FeatureResult] = FeatureResult()
    f_hit_extremist: Optional[FeatureResult] = FeatureResult()
    f_repeat_rapid: Optional[FeatureResult] = FeatureResult()
    f_fresh: Optional[FeatureResult] = FeatureResult()
    
    # --- DIMENSION 5: SPATIAL ---
    f_loc_dist: Optional[FeatureResult] = FeatureResult()
    f_conflict: Optional[FeatureResult] = FeatureResult()
    f_local_sim: Optional[FeatureResult] = FeatureResult()
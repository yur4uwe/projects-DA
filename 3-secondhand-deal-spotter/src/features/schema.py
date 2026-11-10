"""Schema definitions and feature constants for secondary PC marketplace listings."""

TARGET_COL = "deal_tier"

# Lexicographical ordering matching scikit-learn classes_ attribute
DEAL_TIERS = ["Fair Price", "Great Deal", "Overpriced"]

NUMERICAL_FEATURES = [
    "Inches",
    "Ram_GB",
    "Weight_kg",
    "Cpu_Freq_GHz",
    "Primary_Storage_GB",
    "Total_Storage_GB",
    "Screen_Width",
    "Screen_Height",
    "PPI",
    "asking_price",
    "market_baseline_price",
    "price_ratio",
    "price_delta_euros",
]

CATEGORICAL_FEATURES = [
    "Company",
    "TypeName",
    "Cpu_Brand",
    "Cpu_Tier",
    "Primary_Storage_Type",
    "Gpu_Brand",
    "Gpu_Tier",
    "OpSys",
]

BOOLEAN_FEATURES = [
    "Is_Touchscreen",
    "Is_IPS",
    "Has_Dedicated_Gpu",
    "Has_SSD",
]

MODEL_PREDICTOR_NUMERICAL = [
    "Inches",
    "Ram_GB",
    "Weight_kg",
    "Cpu_Freq_GHz",
    "Primary_Storage_GB",
    "Total_Storage_GB",
    "PPI",
    "asking_price",
    "market_baseline_price",
    "price_delta_euros",
]

MODEL_PREDICTOR_CATEGORICAL = CATEGORICAL_FEATURES
MODEL_PREDICTOR_BOOLEAN = BOOLEAN_FEATURES

MODEL_FEATURE_COLUMNS = (
    MODEL_PREDICTOR_NUMERICAL
    + MODEL_PREDICTOR_CATEGORICAL
    + MODEL_PREDICTOR_BOOLEAN
)

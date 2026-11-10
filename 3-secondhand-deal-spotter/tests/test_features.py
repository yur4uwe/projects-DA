"""Unit tests for feature engineering, cleaning, and market baselines."""

import numpy as np
import pandas as pd

from src.features.baseline import MarketBaselineModel, assign_deal_tier
from src.features.cleaner import (
    clean_and_parse_listings,
    clean_opsys,
    parse_cpu_details,
    parse_gpu_details,
    parse_storage_details,
)


def test_parse_cpu_details() -> None:
    brand, tier, speed = parse_cpu_details("Intel Core i7 7700HQ 2.8GHz")
    assert brand == "Intel"
    assert tier == "Core i7"
    assert speed == 2.8

    amd_brand, amd_tier, amd_speed = parse_cpu_details("AMD Ryzen 7 1700 3GHz")
    assert amd_brand == "AMD"
    assert amd_tier == "AMD Ryzen"
    assert amd_speed == 3.0


def test_parse_gpu_details() -> None:
    nvidia_brand, nvidia_tier = parse_gpu_details("Nvidia GeForce GTX 1070")
    assert nvidia_brand == "NVIDIA"
    assert nvidia_tier == "Dedicated Gaming"

    intel_brand, intel_tier = parse_gpu_details("Intel HD Graphics 620")
    assert intel_brand == "Intel"
    assert intel_tier == "Integrated"


def test_parse_storage_details() -> None:
    p_size, p_type, s_size, s_type, total = parse_storage_details(
        "256GB SSD + 1TB HDD"
    )
    assert p_size == 256
    assert p_type == "SSD"
    assert s_size == 1024
    assert s_type == "HDD"
    assert total == 1280


def test_clean_opsys() -> None:
    assert clean_opsys("Windows 10") == "Windows"
    assert clean_opsys("macOS") == "macOS"
    assert clean_opsys("Linux") == "Linux"
    assert clean_opsys("No OS") == "Other / No OS"


def test_clean_and_parse_listings_sample() -> None:
    raw_sample = pd.DataFrame(
        [
            {
                "Company": "Dell",
                "Product": "Inspiron 15",
                "TypeName": "Notebook",
                "Inches": 15.6,
                "ScreenResolution": "Full HD 1920x1080",
                "Cpu": "Intel Core i5 7200U 2.5GHz",
                "Ram": "8GB",
                "Memory": "256GB SSD",
                "Gpu": "Intel HD Graphics 620",
                "OpSys": "Windows 10",
                "Weight": "2.1kg",
                "Price_euros": 650.0,
            }
        ]
    )
    cleaned = clean_and_parse_listings(raw_sample)
    assert len(cleaned) == 1
    assert cleaned["Ram_GB"].iloc[0] == 8
    assert cleaned["Weight_kg"].iloc[0] == 2.1
    assert cleaned["Screen_Width"].iloc[0] == 1920
    assert cleaned["Screen_Height"].iloc[0] == 1080
    assert cleaned["Cpu_Tier"].iloc[0] == "Core i5"
    assert cleaned["Primary_Storage_Type"].iloc[0] == "SSD"
    assert cleaned["Has_SSD"].iloc[0] == True


def test_assign_deal_tier() -> None:
    assert assign_deal_tier(0.70) == "Great Deal"
    assert assign_deal_tier(0.85) == "Great Deal"
    assert assign_deal_tier(1.00) == "Fair Price"
    assert assign_deal_tier(1.15) == "Fair Price"
    assert assign_deal_tier(1.25) == "Overpriced"


def test_market_baseline_model() -> None:
    model = MarketBaselineModel(random_state=42)
    sample_df = pd.DataFrame(
        [
            {
                "Company": "Dell",
                "TypeName": "Notebook",
                "Cpu_Tier": "Core i5",
                "Gpu_Tier": "Integrated",
                "Primary_Storage_Type": "SSD",
                "OpSys": "Windows",
                "Inches": 15.6,
                "Ram_GB": 8,
                "Weight_kg": 2.0,
                "Cpu_Freq_GHz": 2.5,
                "Primary_Storage_GB": 256,
                "Total_Storage_GB": 256,
                "PPI": 141.2,
                "Is_Touchscreen": False,
                "Is_IPS": False,
                "Has_Dedicated_Gpu": False,
                "Has_SSD": True,
            }
        ]
        * 10
    )
    y = pd.Series([600.0] * 10)
    model.fit(sample_df, y)
    preds = model.predict(sample_df)
    assert len(preds) == 10
    assert np.all(preds > 0)

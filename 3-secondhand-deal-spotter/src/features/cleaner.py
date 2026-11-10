"""Data cleaning and technical specification parsing for PC marketplace listings."""

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_cpu_details(cpu_str: str) -> tuple[str, str, float]:
    """Extract CPU Brand, Tier, and Clock Speed (GHz) from CPU text."""
    if not isinstance(cpu_str, str):
        return "Other", "Other", 2.0

    speed_match = re.search(r"([\d\.]+)\s*GHz", cpu_str, re.IGNORECASE)
    speed_ghz = float(speed_match.group(1)) if speed_match else 2.0
    text = cpu_str.lower()

    if "intel" in text:
        brand = "Intel"
    elif "amd" in text:
        brand = "AMD"
    else:
        brand = "Other"

    if "i7" in text:
        tier = "Core i7"
    elif "i5" in text:
        tier = "Core i5"
    elif "i3" in text:
        tier = "Core i3"
    elif "ryzen" in text:
        tier = "AMD Ryzen"
    elif any(token in text for token in ["a9", "a6", "a12", "a10", "a8", "a4", "fx", "e-series"]):
        tier = "AMD A-Series"
    elif any(token in text for token in ["celeron", "pentium", "atom"]):
        tier = "Entry (Celeron/Pentium)"
    elif "xeon" in text:
        tier = "Intel Xeon"
    elif "core m" in text or "m3" in text:
        tier = "Core M"
    else:
        tier = "Other"

    return brand, tier, speed_ghz


def parse_gpu_details(gpu_str: str) -> tuple[str, str]:
    """Extract GPU Brand and Tier (Dedicated Gaming, Dedicated Pro, Integrated)."""
    if not isinstance(gpu_str, str):
        return "Intel", "Integrated"

    text = gpu_str.lower()
    if "nvidia" in text or "geforce" in text:
        brand = "NVIDIA"
        tier = "Dedicated Pro" if "quadro" in text else "Dedicated Gaming"
    elif "amd" in text or "radeon" in text:
        brand = "AMD"
        if any(token in text for token in ["firepro", "pro"]):
            tier = "Dedicated Pro"
        elif any(token in text for token in ["rx", "r5", "r7", "r9", "520", "530", "540"]):
            tier = "Dedicated Gaming"
        else:
            tier = "Integrated"
    elif "intel" in text:
        brand = "Intel"
        tier = "Integrated"
    else:
        brand = "Other"
        tier = "Integrated"

    return brand, tier


def parse_storage_details(memory_str: str) -> tuple[int, str, int, str, int]:
    """Extract Primary Storage (GB & Type), Secondary Storage (GB & Type), and Total GB."""
    if not isinstance(memory_str, str):
        return 256, "SSD", 0, "None", 256

    def parse_single_storage(part: str) -> tuple[int, str]:
        match = re.search(r"([\d\.]+)\s*(GB|TB)", part, re.IGNORECASE)
        if not match:
            return 0, "Unknown"
        val = float(match.group(1))
        unit = match.group(2).upper()
        size_gb = int(val * 1024) if unit == "TB" else int(val)

        p_lower = part.lower()
        if "ssd" in p_lower:
            storage_type = "SSD"
        elif "hdd" in p_lower:
            storage_type = "HDD"
        elif "flash" in p_lower:
            storage_type = "Flash Storage"
        elif "hybrid" in p_lower:
            storage_type = "Hybrid"
        else:
            storage_type = "Unknown"
        return size_gb, storage_type

    parts = memory_str.split("+")
    p_size, p_type = parse_single_storage(parts[0])
    s_size, s_type = parse_single_storage(parts[1]) if len(parts) > 1 else (0, "None")
    total_gb = p_size + s_size

    return p_size, p_type, s_size, s_type, total_gb


def clean_opsys(opsys_str: str) -> str:
    """Standardize Operating System categories."""
    if not isinstance(opsys_str, str):
        return "Other / No OS"
    if "Windows" in opsys_str:
        return "Windows"
    if "Mac" in opsys_str or "macOS" in opsys_str:
        return "macOS"
    if "Linux" in opsys_str:
        return "Linux"
    return "Other / No OS"


def clean_and_parse_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw PC marketplace listings into clean, standardized feature dataframe."""
    initial_count = len(df)
    clean_df = df.copy()

    # Normalize RAM and Weight
    clean_df["Ram_GB"] = clean_df["Ram"].astype(str).str.extract(r"(\d+)").astype(int)
    clean_df["Weight_kg"] = (
        clean_df["Weight"].astype(str).str.extract(r"([\d\.]+)").astype(float)
    )

    # Resolution and Display properties
    res_match = clean_df["ScreenResolution"].astype(str).str.extract(r"(\d{3,4})x(\d{3,4})")
    clean_df["Screen_Width"] = res_match[0].astype(int)
    clean_df["Screen_Height"] = res_match[1].astype(int)
    clean_df["Inches"] = clean_df["Inches"].astype(float)
    clean_df["PPI"] = np.round(
        np.sqrt(clean_df["Screen_Width"] ** 2 + clean_df["Screen_Height"] ** 2)
        / clean_df["Inches"],
        2,
    )
    clean_df["Is_Touchscreen"] = (
        clean_df["ScreenResolution"].astype(str).str.contains("touchscreen", case=False)
    )
    clean_df["Is_IPS"] = clean_df["ScreenResolution"].astype(str).str.contains("ips", case=False)

    # CPU Attributes
    cpu_parsed = clean_df["Cpu"].apply(parse_cpu_details)
    clean_df["Cpu_Brand"] = [c[0] for c in cpu_parsed]
    clean_df["Cpu_Tier"] = [c[1] for c in cpu_parsed]
    clean_df["Cpu_Freq_GHz"] = [c[2] for c in cpu_parsed]

    # GPU Attributes
    gpu_parsed = clean_df["Gpu"].apply(parse_gpu_details)
    clean_df["Gpu_Brand"] = [g[0] for g in gpu_parsed]
    clean_df["Gpu_Tier"] = [g[1] for g in gpu_parsed]
    clean_df["Has_Dedicated_Gpu"] = clean_df["Gpu_Tier"].isin(
        ["Dedicated Gaming", "Dedicated Pro"]
    )

    # Storage Attributes
    storage_parsed = clean_df["Memory"].apply(parse_storage_details)
    clean_df["Primary_Storage_GB"] = [s[0] for s in storage_parsed]
    clean_df["Primary_Storage_Type"] = [s[1] for s in storage_parsed]
    clean_df["Secondary_Storage_GB"] = [s[2] for s in storage_parsed]
    clean_df["Total_Storage_GB"] = [s[4] for s in storage_parsed]
    clean_df["Has_SSD"] = (clean_df["Primary_Storage_Type"] == "SSD") | (
        clean_df["Memory"].astype(str).str.contains("SSD", case=False)
    )

    # Operating System & Pricing
    clean_df["OpSys"] = clean_df["OpSys"].apply(clean_opsys)
    clean_df["asking_price"] = clean_df["Price_euros"].astype(float)

    logger.info("Parsed and cleaned %d records (from initial %d).", len(clean_df), initial_count)
    return clean_df

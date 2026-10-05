import re
from typing import Any

from bs4 import BeautifulSoup

from .schema import CleanedJobRecord
from .skills import extract_skills


def clean_html(text: str) -> str:
    """Remove HTML tags, linebreaks, and collapse whitespace."""
    if not text:
        return ""
    # Strip HTML tags
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_salary(
    base_salary: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """Normalize salary to monthly USD (min, max, mid)."""
    if not base_salary or not isinstance(base_salary, dict):
        return None, None, None

    currency = base_salary.get("currency", "USD")
    if currency != "USD":
        # For now, focus on USD denominated compensation
        return None, None, None

    val = base_salary.get("value", {})
    if isinstance(val, dict):
        raw_min = val.get("minValue")
        raw_max = val.get("maxValue")
        unit = val.get("unitText", "MONTH").upper()
    else:
        raw_min = base_salary.get("minValue")
        raw_max = base_salary.get("maxValue")
        unit = base_salary.get("unitText", "MONTH").upper()

    try:
        s_min = float(raw_min) if raw_min is not None else None
    except (ValueError, TypeError):
        s_min = None

    try:
        s_max = float(raw_max) if raw_max is not None else None
    except (ValueError, TypeError):
        s_max = None

    # Convert yearly compensation (e.g. RemoteOK) to monthly
    if unit in ("YEAR", "ANNUAL", "YEARLY"):
        if s_min is not None:
            s_min /= 12.0
        if s_max is not None:
            s_max /= 12.0

    # Calculate midpoint
    if s_min is not None and s_max is not None:
        s_mid = (s_min + s_max) / 2.0
    elif s_min is not None:
        s_mid = s_min
    elif s_max is not None:
        s_mid = s_max
    else:
        s_mid = None

    # Sanity filter: monthly USD tech compensation threshold ($300 to $35,000 / mo)
    if s_mid is not None and not (300 <= s_mid <= 35000):
        return None, None, None

    return (
        round(s_min, 2) if s_min else None,
        round(s_max, 2) if s_max else None,
        round(s_mid, 2) if s_mid else None,
    )


def normalize_experience(
    exp_req: dict[str, Any], title: str, description: str
) -> float | None:
    """Derive numerical years of experience required."""
    # 1. Check explicit structured months
    if isinstance(exp_req, dict) and exp_req.get("monthsOfExperience") is not None:
        try:
            return round(float(exp_req["monthsOfExperience"]) / 12.0, 1)
        except (ValueError, TypeError):
            pass

    # 2. Check explicit regex in title or description (e.g., "5+ years", "3 years of experience")
    combined_text = f"{title} {description}".lower()
    match = re.search(
        r"(\d+)(?:\+)?\s*(?:years?|роки?|років|лет)\s*(?:of\s*experience)?",
        combined_text,
    )
    if match:
        years = float(match.group(1))
        if 0.5 <= years <= 15:
            return years

    # 3. Fallback based on seniority keyword in title
    title_lower = title.lower()
    if any(
        k in title_lower for k in ("lead", "principal", "architect", "staff", "head")
    ):
        return 6.0
    if any(k in title_lower for k in ("senior", "sr.")):
        return 5.0
    if any(k in title_lower for k in ("middle", "mid-level", "mid")):
        return 2.5
    if any(k in title_lower for k in ("junior", "jr.")):
        return 1.0
    if any(k in title_lower for k in ("trainee", "intern")):
        return 0.0

    return None


def extract_english_level(text: str) -> str | None:
    """Detect required English proficiency level."""
    text_lower = text.lower()
    if re.search(r"\b(c2|fluent|native)\b", text_lower):
        return "Fluent / C2"
    if re.search(r"\b(c1|advanced)\b", text_lower):
        return "Advanced / C1"
    if re.search(r"\b(b2|upper-intermediate|upper intermediate)\b", text_lower):
        return "Upper-Intermediate / B2"
    if re.search(r"\b(b1|intermediate)\b", text_lower):
        return "Intermediate / B1"
    if re.search(r"\b(a2|pre-intermediate)\b", text_lower):
        return "Pre-Intermediate / A2"
    return None


def clean_record(raw: dict[str, Any]) -> CleanedJobRecord:
    """Transform a single raw ingested dictionary into a CleanedJobRecord."""
    title = raw.get("title", "").strip()
    raw_desc = raw.get("description", "")
    desc_cleaned = clean_html(raw_desc)

    s_min, s_max, s_mid = normalize_salary(raw.get("base_salary", {}))
    exp_years = normalize_experience(
        raw.get("experience_requirements", {}), title, desc_cleaned
    )
    english = extract_english_level(desc_cleaned)
    skills = extract_skills(f"{title} {desc_cleaned}", raw.get("tags", []))

    location_type = raw.get("job_location_type", "")
    remote = "Remote" if "telecommute" in location_type.lower() else "On-site/Hybrid"

    return CleanedJobRecord(
        id=str(raw.get("id", "")),
        source=raw.get("source", "unknown"),
        title=title,
        category=raw.get("category", "").strip(),
        hiring_organization=raw.get("hiring_organization", "").strip(),
        experience_years=exp_years,
        salary_min_usd=s_min,
        salary_max_usd=s_max,
        salary_mid_usd=s_mid,
        remote_type=remote,
        english_level=english,
        skills=skills,
        description_cleaned=desc_cleaned,
    )

import re
from typing import Any

from bs4 import BeautifulSoup

from .schema import CleanedJobRecord
from .skills import extract_skills


def clean_html(text: str) -> str:
    """Remove HTML tags, linebreaks, and collapse whitespace."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_salary(
    base_salary: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    """Normalize salary to monthly USD (min, max, mid)."""
    if not base_salary or not isinstance(base_salary, dict):
        return None, None, None

    currency = base_salary.get("currency", "USD")
    if currency != "USD":
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
) -> tuple[float | None, bool]:
    """Derive numerical years of experience and whether it was inferred."""
    # 1. Check explicit structured months
    if isinstance(exp_req, dict) and exp_req.get("monthsOfExperience") is not None:
        try:
            return round(float(exp_req["monthsOfExperience"]) / 12.0, 1), False
        except (ValueError, TypeError):
            pass

    # 2. Check explicit regex in title or description
    combined_text = f"{title} {description}".lower()
    match = re.search(
        r"(\d+)(?:\+)?\s*(?:years?|роки?|років|лет)\s*(?:of\s*experience)?",
        combined_text,
    )
    if match:
        years = float(match.group(1))
        if 0.5 <= years <= 15:
            return years, False

    # 3. Fallback based on seniority keyword in title (inferred)
    title_lower = title.lower()
    if any(
        k in title_lower for k in ("lead", "principal", "architect", "staff", "head")
    ):
        return 6.0, True
    if any(k in title_lower for k in ("senior", "sr.")):
        return 5.0, True
    if any(k in title_lower for k in ("middle", "mid-level", "mid")):
        return 2.5, True
    if any(k in title_lower for k in ("junior", "jr.")):
        return 1.0, True
    if any(k in title_lower for k in ("trainee", "intern")):
        return 0.0, True

    return None, True


def infer_seniority_level(title: str, exp_years: float | None) -> int:
    """Classify into standard ordinal seniority: 0 (Trainee) to 4 (Lead/Staff/Architect)."""
    title_lower = title.lower()
    if any(
        k in title_lower
        for k in (
            "lead",
            "principal",
            "architect",
            "staff",
            "head",
            "director",
            "manager",
        )
    ):
        return 4
    if any(k in title_lower for k in ("senior", "sr.")):
        return 3
    if any(k in title_lower for k in ("middle", "mid-level", "mid")):
        return 2
    if any(k in title_lower for k in ("junior", "jr.")):
        return 1
    if any(k in title_lower for k in ("trainee", "intern")):
        return 0

    if exp_years is not None:
        if exp_years >= 5.5:
            return 4
        if exp_years >= 3.5:
            return 3
        if exp_years >= 1.5:
            return 2
        if exp_years >= 0.5:
            return 1
        return 0

    return 2  # Default to Mid-level if unspecified


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


def get_english_rank(level: str | None) -> int:
    """Convert CEFR label to ordinal integer rank (0 to 4)."""
    ranks = {
        "Pre-Intermediate / A2": 1,
        "Intermediate / B1": 2,
        "Upper-Intermediate / B2": 3,
        "Advanced / C1": 4,
        "Fluent / C2": 4,
    }
    return ranks.get(level or "", 0)


def clean_record(raw: dict[str, Any]) -> CleanedJobRecord:
    """Transform a single raw ingested dictionary into a CleanedJobRecord."""
    title = raw.get("title", "").strip()
    raw_desc = raw.get("description", "")
    desc_cleaned = clean_html(raw_desc)

    s_min, s_max, s_mid = normalize_salary(raw.get("base_salary", {}))
    exp_years, is_inferred = normalize_experience(
        raw.get("experience_requirements", {}), title, desc_cleaned
    )
    seniority = infer_seniority_level(title, exp_years)
    english = extract_english_level(desc_cleaned)
    english_rank = get_english_rank(english)
    skills = extract_skills(f"{title} {desc_cleaned}", raw.get("tags", []))

    location_type = raw.get("job_location_type", "")
    is_remote = (
        1
        if "telecommute" in location_type.lower()
        or "remote" in str(raw.get("tags", [])).lower()
        else 0
    )
    remote_label = "Remote" if is_remote else "On-site/Hybrid"

    return CleanedJobRecord(
        id=str(raw.get("id", "")),
        source=raw.get("source", "unknown"),
        title=title,
        category=raw.get("category", "").strip(),
        hiring_organization=raw.get("hiring_organization", "").strip(),
        experience_years=exp_years,
        is_experience_inferred=is_inferred,
        seniority_level=seniority,
        salary_min_usd=s_min,
        salary_max_usd=s_max,
        salary_mid_usd=s_mid,
        remote_type=remote_label,
        is_remote=is_remote,
        english_level=english,
        english_proficiency_rank=english_rank,
        skills=skills,
        description_cleaned=desc_cleaned,
    )

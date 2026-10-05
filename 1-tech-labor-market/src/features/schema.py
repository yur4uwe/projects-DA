from dataclasses import dataclass, field


@dataclass
class CleanedJobRecord:
    """Standardized schema for a processed job posting."""

    id: str
    source: str
    title: str
    category: str
    hiring_organization: str
    experience_years: float | None
    is_experience_inferred: bool
    seniority_level: int
    salary_min_usd: float | None
    salary_max_usd: float | None
    salary_mid_usd: float | None
    remote_type: str
    is_remote: int
    english_level: str | None
    english_proficiency_rank: int
    role_family: str = "other"
    skills: list[str] = field(default_factory=list)
    description_cleaned: str = ""

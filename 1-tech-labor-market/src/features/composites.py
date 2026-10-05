from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .skills import ALL_TECH_PATTERNS, SKILL_GROUPS


@dataclass
class SkillComposites:
    """Aggregated domain counts, interaction flags, and binary indicators."""

    num_skills: int
    num_languages: int
    num_frontend: int
    num_backend: int
    num_databases: int
    num_cloud: int
    num_devops: int
    num_data_ml: int
    num_genai: int
    num_mobile: int

    # Interaction and domain flags
    is_fullstack: int
    is_cloud_native: int
    is_ai_ml: int
    is_devops_oriented: int

    # Individual binary flags: {"skill_python": 1, ...}
    flags: dict[str, int]


def compute_skill_composites(skills: list[str]) -> SkillComposites:
    """Derive composite features and binary flags from a list of canonical skills."""
    skill_set = set(skills or [])

    # Domain counts
    num_languages = sum(1 for s in skill_set if s in SKILL_GROUPS["languages"])
    num_frontend = sum(1 for s in skill_set if s in SKILL_GROUPS["frontend"])
    num_backend = sum(1 for s in skill_set if s in SKILL_GROUPS["backend"])
    num_databases = sum(1 for s in skill_set if s in SKILL_GROUPS["databases"])
    num_cloud = sum(1 for s in skill_set if s in SKILL_GROUPS["cloud"])
    num_devops = sum(1 for s in skill_set if s in SKILL_GROUPS["devops"])
    num_data_ml = sum(1 for s in skill_set if s in SKILL_GROUPS["data_ml"])
    num_genai = sum(1 for s in skill_set if s in SKILL_GROUPS["genai"])
    num_mobile = sum(1 for s in skill_set if s in SKILL_GROUPS["mobile"])

    # High-level domain flags
    is_fullstack = int(
        num_frontend > 0
        and (num_backend > 0 or "javascript" in skill_set or "typescript" in skill_set)
    )
    is_cloud_native = int(
        num_cloud > 0 and ("docker" in skill_set or "kubernetes" in skill_set)
    )
    is_ai_ml = int(num_data_ml > 0 or num_genai > 0)
    is_devops_oriented = int(
        num_devops >= 2 or ("terraform" in skill_set and "kubernetes" in skill_set)
    )

    # Binary flags for all individual canonical skills
    flags = {
        f"skill_{name.replace('-', '_').replace('/', '_')}": int(name in skill_set)
        for name in ALL_TECH_PATTERNS
    }

    return SkillComposites(
        num_skills=len(skill_set),
        num_languages=num_languages,
        num_frontend=num_frontend,
        num_backend=num_backend,
        num_databases=num_databases,
        num_cloud=num_cloud,
        num_devops=num_devops,
        num_data_ml=num_data_ml,
        num_genai=num_genai,
        num_mobile=num_mobile,
        is_fullstack=is_fullstack,
        is_cloud_native=is_cloud_native,
        is_ai_ml=is_ai_ml,
        is_devops_oriented=is_devops_oriented,
        flags=flags,
    )


def add_skill_composites_to_df(
    df: pd.DataFrame, skills_col: str = "skills"
) -> pd.DataFrame:
    """Expand a DataFrame containing a skills column with all composite features and binary flags."""
    if df.empty or skills_col not in df.columns:
        return df

    composite_rows: list[dict[str, Any]] = []
    for skill_list in df[skills_col]:
        comp = compute_skill_composites(
            skill_list if isinstance(skill_list, list) else []
        )
        row_dict = asdict(comp)
        flags = row_dict.pop("flags")
        row_dict.update(flags)
        composite_rows.append(row_dict)

    composites_df = pd.DataFrame(composite_rows, index=df.index)
    return pd.concat([df, composites_df], axis=1)

import re
from collections.abc import Iterable

# Canonical skill taxonomy grouped by domain
SKILL_GROUPS: dict[str, dict[str, list[str]]] = {
    "languages": {
        "python": [r"\bpython\b", r"\bpython3\b"],
        "go": [r"\bgolang\b", r"\bgo\b"],
        "java": [r"\bjava\b"],
        "kotlin": [r"\bkotlin\b"],
        "csharp": [r"\bc#\b", r"\bcsharp\b", r"\b\.net\b", r"\bdotnet\b"],
        "cpp": [r"\bc\+\+\b", r"\bcpp\b"],
        "c": [r"\bc\b(?!\+\+|#)"],  # C language (excluding C++ and C#)
        "javascript": [r"\bjavascript\b", r"\bjs\b", r"\bes6\b"],
        "typescript": [r"\btypescript\b", r"\bts\b"],
        "ruby": [r"\bruby\b", r"\brails\b"],
        "php": [r"\bphp\b", r"\blaravel\b"],
        "rust": [r"\brust\b"],
        "swift": [r"\bswift\b"],
        "scala": [r"\bscala\b"],
        "sql": [r"\bsql\b"],
    },
    "frontend": {
        "react": [r"\breact\b", r"\breactjs\b", r"\breact\.js\b"],
        "vue": [r"\bvue\b", r"\bvuejs\b", r"\bvue\.js\b"],
        "angular": [r"\bangular\b", r"\bangularjs\b"],
        "nextjs": [r"\bnext\.?js\b", r"\bnextjs\b"],
        "tailwind": [r"\btailwind\b", r"\btailwindcss\b"],
    },
    "backend": {
        "django": [r"\bdjango\b"],
        "fastapi": [r"\bfastapi\b", r"\bfast-api\b"],
        "flask": [r"\bflask\b"],
        "spring": [r"\bspring\b", r"\bspring boot\b"],
        "nodejs": [r"\bnode\.?js\b", r"\bnodejs\b"],
        "graphql": [r"\bgraphql\b"],
        "grpc": [r"\bgrpc\b"],
    },
    "databases": {
        "postgresql": [r"\bpostgres\b", r"\bpostgresql\b"],
        "mysql": [r"\bmysql\b"],
        "mongodb": [r"\bmongodb\b", r"\bmongo\b"],
        "redis": [r"\bredis\b"],
        "elasticsearch": [r"\belasticsearch\b", r"\belastic\b"],
        "kafka": [r"\bkafka\b"],
        "rabbitmq": [r"\brabbitmq\b"],
        "clickhouse": [r"\bclickhouse\b"],
    },
    "cloud": {
        "aws": [r"\baws\b", r"\bamazon web services\b"],
        "gcp": [r"\bgcp\b", r"\bgoogle cloud\b"],
        "azure": [r"\bazure\b"],
    },
    "devops": {
        "docker": [r"\bdocker\b"],
        "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
        "terraform": [r"\bterraform\b"],
        "ci/cd": [r"\bci/cd\b", r"\bgithub actions\b", r"\bgitlab ci\b"],
        "linux": [r"\blinux\b"],
    },
    "data_ml": {
        "pandas": [r"\bpandas\b"],
        "numpy": [r"\bnumpy\b"],
        "scikit-learn": [r"\bscikit-learn\b", r"\bsklearn\b"],
        "pytorch": [r"\bpytorch\b"],
        "tensorflow": [r"\btensorflow\b"],
        "spark": [r"\bspark\b", r"\bpyspark\b"],
        "airflow": [r"\bairflow\b"],
        "dbt": [r"\bdbt\b"],
        "bigquery": [r"\bbigquery\b"],
        "snowflake": [r"\bsnowflake\b"],
        "databricks": [r"\bdatabricks\b"],
        "polars": [r"\bpolars\b"],
    },
    "genai": {
        "llm": [r"\bllm\b", r"\bllms\b", r"\blarge language models?\b"],
        "rag": [r"\brag\b", r"\bretrieval[- ]augmented\b"],
        "langchain": [r"\blangchain\b"],
        "transformers": [r"\btransformers\b", r"\bhuggingface\b", r"\bhugging face\b"],
    },
    "mobile": {
        "flutter": [r"\bflutter\b"],
        "react_native": [r"\breact[- ]native\b"],
        "ios": [r"\bios\b"],
        "android": [r"\bandroid\b"],
    },
}

# Flatten for global matching
ALL_TECH_PATTERNS: dict[str, list[str]] = {
    skill: patterns
    for group in SKILL_GROUPS.values()
    for skill, patterns in group.items()
}

# Precompile regex patterns for performance
COMPILED_PATTERNS = {
    skill: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for skill, patterns in ALL_TECH_PATTERNS.items()
}


def extract_skills(text: str, tags: Iterable[str] = ()) -> list[str]:
    """Extract canonical skills from free text and pre-existing tags."""
    found_skills: set[str] = set()

    # 1. Match from explicit tags
    for tag in tags:
        normalized_tag = tag.strip().lower()
        for skill, patterns in COMPILED_PATTERNS.items():
            if any(p.search(normalized_tag) for p in patterns):
                found_skills.add(skill)

    # 2. Match from text (title + description)
    if text:
        for skill, patterns in COMPILED_PATTERNS.items():
            if skill in found_skills:
                continue
            if any(p.search(text) for p in patterns):
                found_skills.add(skill)

    return sorted(found_skills)

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RoleDefinition:
    key: str
    label: str
    title_pattern: re.Pattern
    category_pattern: re.Pattern | None = None
    default_title_template: str = "{seniority} {role} Developer"
    description: str = ""


ROLE_DEFINITIONS: list[RoleDefinition] = [
    RoleDefinition(
        key="tech_management",
        label="Tech Management & Architecture",
        title_pattern=re.compile(
            r"\b(tech lead|team lead|architect|engineering manager|cto|head of eng|director of eng|vp eng)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Technical Lead / Architect",
        description="Engineering leaders, team leads, software architects, and engineering directors",
    ),
    RoleDefinition(
        key="product_project",
        label="Product & Project Management",
        title_pattern=re.compile(
            r"\b(product manager|project manager|scrum master|delivery manager|product owner)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Technical Product Manager",
        description="Product managers, technical project managers, and scrum masters",
    ),
    RoleDefinition(
        key="devops_cloud",
        label="DevOps, SRE & Cloud Infrastructure",
        title_pattern=re.compile(
            r"\b(devops|sre|site reliability|cloud engineer|platform engineer|infrastructure|sysadmin|system admin)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} DevOps / Cloud Engineer",
        description="Cloud infrastructure, Kubernetes, CI/CD, and site reliability engineering",
    ),
    RoleDefinition(
        key="data_ai_ml",
        label="Data, AI & Machine Learning",
        title_pattern=re.compile(
            r"\b(data engineer|machine learning|ml engineer|data scientist|ai engineer|nlp|data analyst|bi engineer|analytics engineer|data annotator)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Machine Learning / Data Engineer",
        description="Data engineering, analytics, ML model deployment, and AI pipelines",
    ),
    RoleDefinition(
        key="qa_testing",
        label="QA & Test Automation",
        title_pattern=re.compile(
            r"\b(qa|quality assurance|sdet|software test|tester|test automation|manual qa|automation qa)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Automation QA Engineer",
        description="Quality assurance, automated testing frameworks, and SDET",
    ),
    RoleDefinition(
        key="mobile_engineer",
        label="Mobile Development (iOS / Android / Flutter)",
        title_pattern=re.compile(
            r"\b(ios|android|mobile developer|flutter|react native|swift developer|kotlin developer)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Mobile Engineer",
        description="Native iOS, Android, and cross-platform mobile development",
    ),
    RoleDefinition(
        key="embedded_hardware",
        label="Embedded Systems & Hardware",
        title_pattern=re.compile(
            r"\b(embedded|hardware|firmware|stm32|iot|rf engineer|fpga|microcontroller)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Embedded Systems Engineer",
        description="Firmware, microcontrollers (STM32/ESP), IoT, and hardware design",
    ),
    RoleDefinition(
        key="fullstack_engineer",
        label="Full Stack Engineer",
        title_pattern=re.compile(
            r"\b(fullstack|full-stack|full stack)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Full Stack Engineer",
        description="End-to-end web applications combining frontend and backend services",
    ),
    RoleDefinition(
        key="frontend_engineer",
        label="Frontend Engineer",
        title_pattern=re.compile(
            r"\b(frontend|front-end|front end|ui developer|web developer|react developer|vue developer|angular developer)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Frontend Engineer",
        description="Client-side user interfaces, modern JavaScript frameworks, and web apps",
    ),
    RoleDefinition(
        key="backend_engineer",
        label="Backend Engineer",
        title_pattern=re.compile(
            r"\b(backend|back-end|back end|python developer|go developer|golang developer|java developer|node developer|ruby developer|c\+\+ developer|c# developer|\.net developer|php developer|rust developer|salesforce developer|software engineer|developer|programmer|engineer)\b",
            re.IGNORECASE,
        ),
        category_pattern=re.compile(
            r"\b(python|java|javascript|golang|go|\.net|php|rust|c\+\+|c#|ruby)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Backend Software Engineer",
        description="Server-side systems, distributed APIs, microservices, and databases",
    ),
    RoleDefinition(
        key="design_creative",
        label="UI/UX & Product Design",
        title_pattern=re.compile(
            r"\b(designer|ui/ux|ux/ui|creative|2d|3d|motion|animator|illustrator)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Product / UI-UX Designer",
        description="User experience, interface design, graphic design, and 3D assets",
    ),
    RoleDefinition(
        key="marketing_growth",
        label="Marketing, Growth & Media Buying",
        title_pattern=re.compile(
            r"\b(media buyer|marketing|маркетолог|seo|aso|ppc|copywriter|content manager|growth)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Digital Marketing & Growth Specialist",
        description="Performance marketing, user acquisition, media buying, and SEO",
    ),
    RoleDefinition(
        key="sales_support",
        label="Sales, Account Management & Support",
        title_pattern=re.compile(
            r"\b(sales|account manager|lead generation|outreach|customer support|support|bdr|sdr|closer|менеджер з продажу|менеджер по роботі з клієнтами)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} Account & Sales Manager",
        description="Tech sales, business development, customer success, and outreach",
    ),
    RoleDefinition(
        key="hr_recruiting",
        label="HR & Talent Acquisition",
        title_pattern=re.compile(
            r"\b(hr|recruiter|talent|people partner|recruiting)\b",
            re.IGNORECASE,
        ),
        default_title_template="{seniority} IT Recruiter / People Partner",
        description="Technical recruiting, talent operations, and HR people partnership",
    ),
]

ROLE_FAMILY_LABELS: dict[str, str] = {r.key: r.label for r in ROLE_DEFINITIONS}
ROLE_FAMILY_LABELS["other"] = "Other / General Tech"

# Reverse lookup for UI dropdowns: Label -> key
ROLE_FAMILY_CHOICES: dict[str, str] = {r.label: r.key for r in ROLE_DEFINITIONS}
ROLE_FAMILY_CHOICES["Other / General Tech"] = "other"


def extract_role_family(title: str, category: str = "") -> str:
    """Classify vacancy into standardized market role family using compiled hierarchical patterns."""
    combined = f"{title} {category}"
    for role_def in ROLE_DEFINITIONS:
        if role_def.title_pattern.search(combined):
            return role_def.key
        if role_def.category_pattern and role_def.category_pattern.search(category):
            return role_def.key
    return "other"


def get_role_label(role_key: str) -> str:
    """Retrieve human-friendly label for a role key."""
    return ROLE_FAMILY_LABELS.get(role_key, role_key.replace("_", " ").title())

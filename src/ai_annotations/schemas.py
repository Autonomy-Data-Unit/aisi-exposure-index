"""Pydantic schemas for the AI job-ad annotation passes.

Each schema validates the LLM output for one pass. The same model is used both as
the structured-output decoder constraint (via ``model_json_schema()``) and as the
parse-time validator (via ``model_validate_json`` or ``extract_json``).

Shared metadata fields (``ad_id``, ``model_name``, ``prompt_version``,
``run_timestamp``, ``parse_success``) are stamped by the export code, not by the
LLM, so they are NOT part of these schemas.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pass 1: Boolean AI Detection
# ---------------------------------------------------------------------------

class BooleanAIDetectionV1(BaseModel):
    """Pass 1 LLM output: nine boolean AI-salience flags + confidence + evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mentions_ai_anywhere: bool
    mentions_genai_or_llm: bool
    mentions_ml_or_data_science_ai: bool
    mentions_chatbot_or_conversational_ai: bool
    mentions_ai_governance_or_risk: bool
    mentions_ai_tool_use_by_worker: bool
    mentions_building_or_maintaining_ai: bool
    mentions_ai_product_or_company_domain: bool
    mentions_automation_of_work: bool
    boolean_pass_confidence: float = Field(ge=0.0, le=1.0)
    boolean_pass_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass 5: Seniority and Management Level
# ---------------------------------------------------------------------------

SeniorityLevel = Literal[
    "intern_or_apprentice",
    "graduate_or_entry_level",
    "junior",
    "mid_level",
    "senior",
    "lead_or_principal",
    "director_or_executive",
    "unclear",
]

ManagementLevel = Literal[
    "no_management",
    "team_lead_or_supervisor",
    "project_or_product_manager",
    "line_manager",
    "department_or_function_head",
    "director_or_executive",
    "unclear",
]


class SeniorityManagementV1(BaseModel):
    """Pass 5 LLM output: seniority level + management level + confidence + evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    seniority_level: SeniorityLevel
    management_level: ManagementLevel
    seniority_management_confidence: float = Field(ge=0.0, le=1.0)
    seniority_management_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass 2: AI Mention Context
# ---------------------------------------------------------------------------

AIMentionContext = Literal[
    "no_ai_mention",
    "generic_employer_boilerplate",
    "worker_expected_to_use_ai_tool",
    "ai_skill_or_experience_requested",
    "role_builds_or_maintains_ai_systems",
    "role_related_to_ai_product_or_service",
    "role_related_to_ai_governance_risk_or_compliance",
    "ambiguous",
]


class AIMentionContextV1(BaseModel):
    """Pass 2 LLM output: why AI is mentioned, single categorical."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ai_mention_context: AIMentionContext
    ai_mention_context_confidence: float = Field(ge=0.0, le=1.0)
    ai_mention_context_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass 3: AI Requirement Level and Kind
# ---------------------------------------------------------------------------

AIRequirementLevel = Literal[
    "none",
    "mentioned_but_not_required",
    "desirable",
    "required",
    "central_to_role",
    "unclear",
]

AIRequirementKind = Literal[
    "none",
    "classical_ml_or_data_science",
    "generative_ai_or_llm",
    "chatbot_or_conversational_ai",
    "ai_engineering_or_mlops",
    "ai_governance_or_risk",
    "ai_product_knowledge",
    "general_ai_literacy",
    "ambiguous_or_other",
]


class AIRequirementV1(BaseModel):
    """Pass 3 LLM output: AI requirement level + kind + confidence + evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ai_requirement_level: AIRequirementLevel
    ai_requirement_kind: AIRequirementKind
    ai_requirement_confidence: float = Field(ge=0.0, le=1.0)
    ai_requirement_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass 4: Worker AI Tool Use and Task Area
# ---------------------------------------------------------------------------

WorkerAIToolUseLevel = Literal[
    "none",
    "possible_or_generic",
    "expected",
    "central_to_work",
    "unclear",
]

WorkerAIToolUseArea = Literal[
    "none",
    "writing_or_editing",
    "search_or_summarisation",
    "data_analysis_or_reporting",
    "software_development",
    "customer_support_or_chat",
    "marketing_content_or_creative",
    "administration_or_operations",
    "education_or_training",
    "legal_compliance_or_review",
    "healthcare_or_clinical_support",
    "management_or_coordination",
    "other",
    "unclear",
]


class WorkerAIToolUseV1(BaseModel):
    """Pass 4 LLM output: worker AI tool use level + task area + confidence + evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    worker_ai_tool_use_level: WorkerAIToolUseLevel
    worker_ai_tool_use_area: WorkerAIToolUseArea
    worker_ai_tool_use_confidence: float = Field(ge=0.0, le=1.0)
    worker_ai_tool_use_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Pass 6: Job-ad Quality, Remote Status, Recruitment Agency
# ---------------------------------------------------------------------------

JobTextInformativeness = Literal[
    "empty_or_title_only",
    "thin",
    "moderate",
    "rich",
]

RemoteStatus = Literal[
    "onsite",
    "hybrid",
    "remote",
    "field_based",
    "unclear",
]


class QualityRemoteAgencyV1(BaseModel):
    """Pass 6 LLM output: text informativeness + remote status + agency flag + confidence + evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    job_text_informativeness: JobTextInformativeness
    remote_status: RemoteStatus
    recruitment_agency_likely: bool
    data_quality_confidence: float = Field(ge=0.0, le=1.0)
    data_quality_evidence: Optional[str] = None

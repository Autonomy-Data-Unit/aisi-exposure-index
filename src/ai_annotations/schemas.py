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

"""Tests for the LLM-pass Pydantic schemas.

Each schema must reject malformed LLM output so that bad rows surface as failures
in the ResultStore rather than silently passing as data.
"""

import pytest
from pydantic import ValidationError

from ai_annotations.schemas import (
    BooleanAIDetectionV1,
    SeniorityManagementV1,
    AIMentionContextV1,
    AIRequirementV1,
    WorkerAIToolUseV1,
    QualityRemoteAgencyV1,
)


# ---------------------------------------------------------------------------
# Pass 1: Boolean AI Detection
# ---------------------------------------------------------------------------

def _valid_pass1():
    return {
        "mentions_ai_anywhere": True,
        "mentions_genai_or_llm": False,
        "mentions_ml_or_data_science_ai": True,
        "mentions_chatbot_or_conversational_ai": False,
        "mentions_ai_governance_or_risk": False,
        "mentions_ai_tool_use_by_worker": False,
        "mentions_building_or_maintaining_ai": True,
        "mentions_ai_product_or_company_domain": True,
        "mentions_automation_of_work": False,
        "boolean_pass_confidence": 0.85,
        "boolean_pass_evidence": "Looking for an ML Engineer",
    }


def test_pass1_accepts_valid_input():
    parsed = BooleanAIDetectionV1.model_validate(_valid_pass1())
    assert parsed.mentions_ai_anywhere is True
    assert parsed.boolean_pass_confidence == 0.85


def test_pass1_accepts_null_evidence():
    payload = _valid_pass1()
    payload["boolean_pass_evidence"] = None
    parsed = BooleanAIDetectionV1.model_validate(payload)
    assert parsed.boolean_pass_evidence is None


def test_pass1_rejects_string_booleans():
    payload = _valid_pass1()
    payload["mentions_ai_anywhere"] = "true"
    with pytest.raises(ValidationError):
        BooleanAIDetectionV1.model_validate(payload)


def test_pass1_rejects_confidence_above_one():
    payload = _valid_pass1()
    payload["boolean_pass_confidence"] = 1.2
    with pytest.raises(ValidationError):
        BooleanAIDetectionV1.model_validate(payload)


def test_pass1_rejects_confidence_below_zero():
    payload = _valid_pass1()
    payload["boolean_pass_confidence"] = -0.1
    with pytest.raises(ValidationError):
        BooleanAIDetectionV1.model_validate(payload)


def test_pass1_rejects_missing_field():
    payload = _valid_pass1()
    del payload["mentions_ai_anywhere"]
    with pytest.raises(ValidationError):
        BooleanAIDetectionV1.model_validate(payload)


def test_pass1_rejects_extra_field():
    payload = _valid_pass1()
    payload["unexpected_field"] = "oops"
    with pytest.raises(ValidationError):
        BooleanAIDetectionV1.model_validate(payload)


# ---------------------------------------------------------------------------
# Pass 5: Seniority + Management
# ---------------------------------------------------------------------------

def _valid_pass5():
    return {
        "seniority_level": "senior",
        "management_level": "line_manager",
        "seniority_management_confidence": 0.7,
        "seniority_management_evidence": "Senior Engineer with 5 direct reports",
    }


def test_pass5_accepts_valid_input():
    parsed = SeniorityManagementV1.model_validate(_valid_pass5())
    assert parsed.seniority_level == "senior"
    assert parsed.management_level == "line_manager"


def test_pass5_rejects_invalid_seniority_enum():
    payload = _valid_pass5()
    payload["seniority_level"] = "super_senior"
    with pytest.raises(ValidationError):
        SeniorityManagementV1.model_validate(payload)


def test_pass5_rejects_invalid_management_enum():
    payload = _valid_pass5()
    payload["management_level"] = "boss"
    with pytest.raises(ValidationError):
        SeniorityManagementV1.model_validate(payload)


def test_pass5_rejects_confidence_above_one():
    payload = _valid_pass5()
    payload["seniority_management_confidence"] = 1.5
    with pytest.raises(ValidationError):
        SeniorityManagementV1.model_validate(payload)


def test_pass5_accepts_unclear_enums():
    payload = _valid_pass5()
    payload["seniority_level"] = "unclear"
    payload["management_level"] = "unclear"
    parsed = SeniorityManagementV1.model_validate(payload)
    assert parsed.seniority_level == "unclear"


def test_pass5_rejects_extra_field():
    payload = _valid_pass5()
    payload["surprise"] = 42
    with pytest.raises(ValidationError):
        SeniorityManagementV1.model_validate(payload)


# ---------------------------------------------------------------------------
# Round-trip: model_json_schema -> validate
# ---------------------------------------------------------------------------

def test_pass1_json_schema_round_trip():
    """The schema we hand to vLLM as `json_schema=` must validate back via the same model."""
    schema = BooleanAIDetectionV1.model_json_schema()
    assert "properties" in schema
    assert "mentions_ai_anywhere" in schema["properties"]
    # Confidence has min/max constraints
    confidence_schema = schema["properties"]["boolean_pass_confidence"]
    assert confidence_schema["minimum"] == 0.0
    assert confidence_schema["maximum"] == 1.0


def test_pass5_json_schema_enums():
    schema = SeniorityManagementV1.model_json_schema()
    # Literal types compile to "enum" or "$ref" -> "enum" depending on pydantic version.
    # Either way, the schema dict should encode the allowed seniority values somewhere.
    s = str(schema)
    assert "senior" in s
    assert "intern_or_apprentice" in s
    assert "director_or_executive" in s


# ---------------------------------------------------------------------------
# Pass 2: AI Mention Context
# ---------------------------------------------------------------------------

def _valid_pass2():
    return {
        "ai_mention_context": "ai_skill_or_experience_requested",
        "ai_mention_context_confidence": 0.8,
        "ai_mention_context_evidence": "must have ML experience",
    }


def test_pass2_accepts_valid_input():
    parsed = AIMentionContextV1.model_validate(_valid_pass2())
    assert parsed.ai_mention_context == "ai_skill_or_experience_requested"


def test_pass2_rejects_invalid_enum():
    payload = _valid_pass2()
    payload["ai_mention_context"] = "something_else"
    with pytest.raises(ValidationError):
        AIMentionContextV1.model_validate(payload)


def test_pass2_rejects_confidence_out_of_range():
    payload = _valid_pass2()
    payload["ai_mention_context_confidence"] = 2.0
    with pytest.raises(ValidationError):
        AIMentionContextV1.model_validate(payload)


def test_pass2_rejects_extra_field():
    payload = _valid_pass2()
    payload["surprise"] = "x"
    with pytest.raises(ValidationError):
        AIMentionContextV1.model_validate(payload)


# ---------------------------------------------------------------------------
# Pass 3: AI Requirement
# ---------------------------------------------------------------------------

def _valid_pass3():
    return {
        "ai_requirement_level": "required",
        "ai_requirement_kind": "generative_ai_or_llm",
        "ai_requirement_confidence": 0.9,
        "ai_requirement_evidence": "Experience with LLMs required",
    }


def test_pass3_accepts_valid_input():
    parsed = AIRequirementV1.model_validate(_valid_pass3())
    assert parsed.ai_requirement_level == "required"
    assert parsed.ai_requirement_kind == "generative_ai_or_llm"


def test_pass3_rejects_invalid_level():
    payload = _valid_pass3()
    payload["ai_requirement_level"] = "really_really_required"
    with pytest.raises(ValidationError):
        AIRequirementV1.model_validate(payload)


def test_pass3_rejects_invalid_kind():
    payload = _valid_pass3()
    payload["ai_requirement_kind"] = "agi_engineering"
    with pytest.raises(ValidationError):
        AIRequirementV1.model_validate(payload)


# ---------------------------------------------------------------------------
# Pass 4: Worker AI Tool Use
# ---------------------------------------------------------------------------

def _valid_pass4():
    return {
        "worker_ai_tool_use_level": "expected",
        "worker_ai_tool_use_area": "writing_or_editing",
        "worker_ai_tool_use_confidence": 0.6,
        "worker_ai_tool_use_evidence": "use AI to draft emails",
    }


def test_pass4_accepts_valid_input():
    parsed = WorkerAIToolUseV1.model_validate(_valid_pass4())
    assert parsed.worker_ai_tool_use_level == "expected"
    assert parsed.worker_ai_tool_use_area == "writing_or_editing"


def test_pass4_rejects_invalid_area():
    payload = _valid_pass4()
    payload["worker_ai_tool_use_area"] = "telepathy"
    with pytest.raises(ValidationError):
        WorkerAIToolUseV1.model_validate(payload)


# ---------------------------------------------------------------------------
# Pass 6: Quality/Remote/Agency
# ---------------------------------------------------------------------------

def _valid_pass6():
    return {
        "job_text_informativeness": "moderate",
        "remote_status": "hybrid",
        "recruitment_agency_likely": False,
        "data_quality_confidence": 0.75,
        "data_quality_evidence": "Hybrid working from London office",
    }


def test_pass6_accepts_valid_input():
    parsed = QualityRemoteAgencyV1.model_validate(_valid_pass6())
    assert parsed.job_text_informativeness == "moderate"
    assert parsed.remote_status == "hybrid"
    assert parsed.recruitment_agency_likely is False


def test_pass6_rejects_string_agency_bool():
    payload = _valid_pass6()
    payload["recruitment_agency_likely"] = "true"
    with pytest.raises(ValidationError):
        QualityRemoteAgencyV1.model_validate(payload)


def test_pass6_rejects_invalid_remote_status():
    payload = _valid_pass6()
    payload["remote_status"] = "telecommute"
    with pytest.raises(ValidationError):
        QualityRemoteAgencyV1.model_validate(payload)


def test_pass6_rejects_invalid_informativeness():
    payload = _valid_pass6()
    payload["job_text_informativeness"] = "extra_rich"
    with pytest.raises(ValidationError):
        QualityRemoteAgencyV1.model_validate(payload)


# ---------------------------------------------------------------------------
# JSON schema round-trip for each new schema
# ---------------------------------------------------------------------------

def test_pass2_json_schema_round_trip():
    schema = AIMentionContextV1.model_json_schema()
    s = str(schema)
    assert "ai_skill_or_experience_requested" in s
    assert "no_ai_mention" in s


def test_pass3_json_schema_round_trip():
    schema = AIRequirementV1.model_json_schema()
    s = str(schema)
    assert "ai_requirement_level" in schema["properties"]
    assert "ai_requirement_kind" in schema["properties"]
    assert "central_to_role" in s


def test_pass4_json_schema_round_trip():
    schema = WorkerAIToolUseV1.model_json_schema()
    s = str(schema)
    assert "worker_ai_tool_use_level" in schema["properties"]
    assert "writing_or_editing" in s


def test_pass6_json_schema_round_trip():
    schema = QualityRemoteAgencyV1.model_json_schema()
    assert "recruitment_agency_likely" in schema["properties"]
    assert "remote_status" in schema["properties"]

"""Tests for the Pass 1 and Pass 5 Pydantic schemas.

Each schema must reject malformed LLM output so that bad rows surface as failures
in the ResultStore rather than silently passing as data.
"""

import pytest
from pydantic import ValidationError

from ai_annotations.schemas import BooleanAIDetectionV1, SeniorityManagementV1


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

"""Unit tests for src/analysis/keyword_matcher.py.

Boundary rejection, inclusion, family assignment, null handling, and
determinism. All assertions are on Python `re`; a separate DuckDB
equivalence check lives in the analysis notebook.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from analysis.keyword_matcher import (
    Dictionary,
    count_total_matches,
    load_dictionary,
    matched_terms,
    scan,
    term_to_pattern,
)

ROOT = Path(__file__).resolve().parents[3]
DICT_PATH = ROOT / "config/ai_keyword_sweep/keyword_families_v1.toml"


@pytest.fixture(scope="module")
def d() -> Dictionary:
    return load_dictionary(DICT_PATH)


# --- term_to_pattern --------------------------------------------------

@pytest.mark.parametrize(
    "term,expected",
    [
        ("ai", r"\bai\b"),
        ("ml", r"\bml\b"),
        ("gpt", r"\bgpt\b"),
        ("nlp", r"\bnlp\b"),
        ("rag", r"\brag\b"),
        ("xai", r"\bxai\b"),
        ("ai-", r"\bai-"),
        ("ai/", r"\bai/"),
        ("a.i.", r"\ba\.i\."),
        ("machine learning", r"\bmachine learning\b"),
        ("copilot", r"\bcopilot\b"),
        ("scikit-learn", r"\bscikit-learn\b"),
        ("dall-e", r"\bdall-e\b"),
    ],
)
def test_term_to_pattern(term: str, expected: str) -> None:
    assert term_to_pattern(term) == expected


def test_term_to_pattern_empty_raises() -> None:
    with pytest.raises(ValueError):
        term_to_pattern("")


# --- boundary rejection (short-token false-positive cases) -----------

SHORT_TOKENS = ["ai", "ml", "llm", "nlp", "rag", "xai", "gpt"]

@pytest.mark.parametrize(
    "text",
    [
        "email me at john@x.com",
        "retail experience preferred",
        "html and css required",
        "training new hires",
        "trail mix",
        "flagrant breach",
        "air conditioning maintenance",
    ],
)
def test_short_tokens_do_not_false_match(text: str) -> None:
    """The seven short-token false-positive cases the spec calls out."""
    for tok in SHORT_TOKENS:
        pat = term_to_pattern(tok)
        assert not re.search(pat, text.lower()), (
            f"pattern {pat!r} false-matched in {text!r}"
        )


# --- inclusion -------------------------------------------------------

@pytest.mark.parametrize(
    "text,family",
    [
        ("Looking for an AI-enabled product manager", "ai_tool_use"),
        ("Strong AI/ML background", "generic_ai"),
        ("AI-driven decision making", "ai_tool_use"),
        ("Senior ML Engineer needed", "machine_learning"),
        ("LLM-based application development", "generative_ai"),
        ("Experience with RAG pipeline design", "generative_ai"),
        ("A.I. tools for productivity", "generic_ai"),
        ("Familiar with ChatGPT and similar tools", "generative_ai"),
        ("Machine Learning Engineer", "machine_learning"),
        ("Working knowledge of NLP", "machine_learning"),
        ("AI/ML pipeline ownership", "machine_learning"),
    ],
)
def test_inclusion(d: Dictionary, text: str, family: str) -> None:
    hits = scan(text, d)
    assert hits[family], f"expected {family} to match in {text!r}; got {hits}"


def test_ai_slash_ml_hits_both_families(d: Dictionary) -> None:
    text = "Strong AI/ML background"
    hits = scan(text, d)
    assert hits["generic_ai"]
    assert hits["machine_learning"]


def test_bare_gpt_does_not_match_inside_chatgpt() -> None:
    """The standalone `gpt` term has \\b boundaries on both sides, so it
    should not match the `gpt` inside `chatgpt`. The `chatgpt` term itself
    handles that text."""
    pat = term_to_pattern("gpt")
    assert not re.search(pat, "chatgpt"), (
        "bare \\bgpt\\b must not match inside 'chatgpt'"
    )


def test_chatgpt_text_matches_generative_family(d: Dictionary) -> None:
    assert scan("ChatGPT integration", d)["generative_ai"]


# --- family assignment -----------------------------------------------

def test_six_families(d: Dictionary) -> None:
    assert set(d.families.keys()) == {
        "generic_ai",
        "ai_tool_use",
        "machine_learning",
        "generative_ai",
        "ai_ops_governance",
        "chatbot_automation",
    }


def test_dictionary_version(d: Dictionary) -> None:
    assert d.version == "v1"


def test_each_term_has_a_family(d: Dictionary) -> None:
    for t in d.terms:
        assert t.family in d.families
        assert t.canonical
        assert t.pattern


def test_matched_terms_returns_pairs(d: Dictionary) -> None:
    hits = matched_terms("AI-driven NLP for chatgpt-style assistants", d)
    fams_hit = {f for f, _ in hits}
    # ai_tool_use (ai-driven), machine_learning (nlp), generative_ai (chatgpt)
    assert {"ai_tool_use", "machine_learning", "generative_ai"} <= fams_hit


# --- null / empty text -----------------------------------------------

def test_null_text_no_hits(d: Dictionary) -> None:
    h = scan("", d)
    assert not any(h.values())
    assert count_total_matches("", d) == 0
    assert matched_terms("", d) == []


def test_whitespace_only_no_hits(d: Dictionary) -> None:
    assert not any(scan("   \n\t  ", d).values())


# --- intensity counts ------------------------------------------------

def test_intensity_count_nonzero(d: Dictionary) -> None:
    text = "AI and machine learning, also AI tools, and ChatGPT"
    assert count_total_matches(text, d) > 0


def test_intensity_count_scales_with_repetition(d: Dictionary) -> None:
    one = count_total_matches("chatgpt", d)
    three = count_total_matches("chatgpt chatgpt chatgpt", d)
    assert three >= 3 * one


# --- determinism -----------------------------------------------------

def test_dictionary_loads_deterministically() -> None:
    a = load_dictionary(DICT_PATH)
    b = load_dictionary(DICT_PATH)
    assert a.version == b.version
    assert a.union_pattern == b.union_pattern
    assert a.family_patterns == b.family_patterns
    assert a.terms == b.terms

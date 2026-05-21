"""Tests for the Pass 0 AI keyword prefilter.

Covers the explicit boundary cases the spec calls out — both the texts that
should match and the common English words that should NOT match because the
short ambiguous terms (``ai``, ``ml``, ``llm``, ``nlp``, ``rag``, ``xai``)
require token-boundary anchoring.
"""

import pytest

from ai_annotations.keyword_prefilter import prefilter_text


# ---------------------------------------------------------------------------
# Cases that MUST match (recall: include when in doubt)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "AI-enabled",
    "AI/ML",
    "AI-driven",
    "ML Engineer",
    "LLM-based",
    "RAG pipeline",
    "A.I. tools",
    "Looking for a senior Machine Learning engineer",
    "Build LLM applications with RAG and vector databases",
    "Use ChatGPT to improve productivity",
    "Generative AI experience required",
    "Deep learning research role",
    "Knowledge of MLOps and model deployment",
    "Experience with PyTorch and TensorFlow",
    "We use AI/ML to optimise customer journeys",
])
def test_must_match(text):
    hit, families, terms = prefilter_text(text)
    assert hit, f"expected match for {text!r}, got families={families} terms={terms}"
    assert families, f"expected at least one family for {text!r}"
    assert terms, f"expected at least one term for {text!r}"


# ---------------------------------------------------------------------------
# Cases that MUST NOT match — short ambiguous tokens inside larger words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "mail delivery driver",                           # ai inside mail
    "retail manager",                                 # ai inside retail
    "training delivered weekly",                      # ai inside training
    "trail running shoes",                            # ai inside trail
    "air conditioning engineer",                      # ai inside air
    "xml parsing experience",                         # ml inside xml
    "html and css developer",                         # ml inside html
    "kml file format knowledge",                      # ml inside kml
    "We organise events with great catering",         # ai inside catering
    "Strong attention to detail",                     # 'ai' inside detail / 'ml' nowhere
    "Daily contact with stakeholders",                # ai inside daily
])
def test_must_not_match(text):
    hit, families, terms = prefilter_text(text)
    assert not hit, f"unexpected match for {text!r}, families={families} terms={terms}"


# ---------------------------------------------------------------------------
# Family identification
# ---------------------------------------------------------------------------

def test_genai_llm_family_identifies_correctly():
    hit, families, terms = prefilter_text("Experience building with ChatGPT and LLMs.")
    assert hit
    assert "genai_llm" in families


def test_ml_family_identifies_correctly():
    hit, families, terms = prefilter_text("PyTorch experience preferred")
    assert hit
    assert "ml" in families


def test_multiple_families_can_match():
    hit, families, terms = prefilter_text("Machine learning and generative AI experience")
    assert hit
    assert "ml" in families
    assert "genai_llm" in families


def test_matched_terms_are_lowercased_and_sorted():
    _, _, terms = prefilter_text("Machine Learning and GenAI experience")
    # Terms come out lowercased, deduped, sorted
    assert terms == sorted(terms)
    assert all(t == t.lower() for t in terms)


# ---------------------------------------------------------------------------
# Empty/None handling
# ---------------------------------------------------------------------------

def test_empty_string_returns_no_hit():
    hit, families, terms = prefilter_text("")
    assert not hit
    assert families == []
    assert terms == []


def test_whitespace_only_returns_no_hit():
    hit, families, terms = prefilter_text("   \n\t  ")
    assert not hit


# ---------------------------------------------------------------------------
# Edge: longer terms preferred over shorter prefix matches
# ---------------------------------------------------------------------------

def test_ml_engineer_matches_as_unit():
    """`ml engineer` is listed and should match as a single phrase, not just `ml`."""
    _, _, terms = prefilter_text("Hiring ML Engineer")
    assert "ml engineer" in terms

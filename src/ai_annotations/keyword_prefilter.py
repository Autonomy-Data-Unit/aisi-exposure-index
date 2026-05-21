"""Pass 0: deterministic AI keyword prefilter.

Boundary-aware regex over lowercased ``title + "\\n" + description``. The filter is
deliberately permissive — recall is the goal. The Pass 1 LLM handles precision.

Boundary semantics: each term is anchored by ``(?<![a-z0-9])`` and ``(?![a-z0-9])`` so
the match cannot extend inside a longer alphanumeric token. This rejects ``mail``
matching ``ai``, ``training`` matching ``ai``, ``xml`` matching ``ml``, etc., while
still matching token-adjacent punctuation like ``AI/ML``, ``AI-enabled``, ``LLM-based``,
or ``A.I.``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

KEYWORD_FAMILIES: Dict[str, List[str]] = {
    "generic_ai": [
        "artificial intelligence",
        "ai",
        "a.i.",
        "algorithmic",
        "intelligent automation",
    ],
    "ai_tool_use": [
        "ai tool",
        "ai tools",
        "ai-assisted",
        "ai assisted",
        "ai-powered",
        "ai powered",
        "ai-enabled",
        "ai enabled",
        "ai-driven",
        "ai driven",
        "ai-based",
        "ai based",
        "using ai",
        "with ai",
        "leverage ai",
        "ai capabilities",
        "ai solutions",
        "ai features",
        "intelligent document",
        "semantic search",
    ],
    "ml": [
        "machine learning",
        "ml",
        "ml engineer",
        "deep learning",
        "predictive model",
        "predictive modelling",
        "model training",
        "classification model",
        "recommendation system",
        "recommender",
        "computer vision",
        "natural language processing",
        "nlp",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "sklearn",
        "keras",
        "xgboost",
        "spacy",
        "transformer",
        "fine-tune",
        "fine-tuning",
        "embeddings",
        "foundation model",
        "multimodal",
        "diffusion model",
        "reinforcement learning",
        "rlhf",
    ],
    "genai_llm": [
        "generative ai",
        "genai",
        "large language model",
        "llm",
        "llms",
        "gpt",
        "chatgpt",
        "claude",
        "gemini",
        "bard",
        "llama",
        "mistral",
        "perplexity",
        "midjourney",
        "dall-e",
        "stable diffusion",
        "copilot",
        "github copilot",
        "microsoft copilot",
        "openai",
        "anthropic",
        "hugging face",
        "huggingface",
        "bedrock",
        "azure openai",
        "vertex ai",
        "cursor",
        "prompt engineer",
        "prompt engineering",
        "rag",
        "retrieval augmented generation",
        "vector database",
        "langchain",
        "llamaindex",
        "ai agent",
        "agentic",
    ],
    "ai_ops_governance": [
        "mlops",
        "model deployment",
        "model monitoring",
        "model evaluation",
        "ai safety",
        "responsible ai",
        "ai governance",
        "algorithmic bias",
        "model risk",
        "explainable ai",
        "xai",
    ],
    "chatbots_automation": [
        "chatbot",
        "virtual assistant",
        "conversational ai",
        "automated decision",
        "intelligent assistant",
    ],
}


def _compile_family(terms: List[str]) -> re.Pattern:
    """Compile one alternation regex with boundary anchors for a family of terms.

    Terms are sorted longest-first so longer terms match before their prefixes
    (e.g. ``ml engineer`` matches as a single hit, not ``ml`` + ``engineer``).
    """
    sorted_terms = sorted(set(terms), key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in sorted_terms)
    pattern = r"(?<![a-z0-9])(?:" + alternation + r")(?![a-z0-9])"
    return re.compile(pattern, re.IGNORECASE)


_COMPILED_FAMILIES: Dict[str, re.Pattern] = {
    name: _compile_family(terms) for name, terms in KEYWORD_FAMILIES.items()
}


def prefilter_text(text: str) -> Tuple[bool, List[str], List[str]]:
    """Run the AI keyword prefilter on a single text.

    Args:
        text: Raw text (typically ``title + "\\n" + description``). Will be
            lowercased internally.

    Returns:
        ``(hit, families, terms)``:
        - ``hit``: True if any family matched.
        - ``families``: Sorted list of family names that produced at least one match.
        - ``terms``: Sorted, deduplicated list of matched canonical terms (lowercased).
    """
    if not text:
        return False, [], []

    lowered = text.lower()
    hit_families: set[str] = set()
    hit_terms: set[str] = set()

    for family_name, pattern in _COMPILED_FAMILIES.items():
        matches = pattern.findall(lowered)
        if matches:
            hit_families.add(family_name)
            hit_terms.update(matches)

    return bool(hit_families), sorted(hit_families), sorted(hit_terms)

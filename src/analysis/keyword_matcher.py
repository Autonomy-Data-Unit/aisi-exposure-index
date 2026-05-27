"""Keyword matcher for the AI keyword sweep.

Loads a TOML dictionary of keyword families and compiles boundary-anchored
regexes that work in both Python `re` and DuckDB regexp_matches (RE2). The
notebook (nbs/analysis/04_keyword_demand_analysis.ipynb) executes the bulk
sweep inside DuckDB using the pattern strings produced here; the same
strings drive the unit tests.

Term-to-pattern rule (from the implementation brief):

    1. Escape regex metacharacters inside the term. In v1 only '.' appears.
    2. Prepend \\b iff the first character is alphanumeric.
    3. Append  \\b iff the last  character is alphanumeric.
    4. If the entry carries `regex_override`, that pattern is used verbatim.

Anchors are only \\b, never lookaround, so the same string is valid RE2.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

_METACHARS = set(r".^$*+?{}[]\|()")


@dataclass(frozen=True)
class Term:
    canonical: str
    family: str
    pattern: str


@dataclass(frozen=True)
class Dictionary:
    version: str
    families: dict[str, str]
    terms: tuple[Term, ...]
    family_patterns: dict[str, str]
    union_pattern: str

    def family_keys(self) -> list[str]:
        return list(self.families.keys())


def term_to_pattern(term: str) -> str:
    """Apply the spec's term-to-pattern rule.

    >>> term_to_pattern("ai")
    '\\\\bai\\\\b'
    >>> term_to_pattern("a.i.")
    '\\\\ba\\\\.i\\\\.'
    >>> term_to_pattern("ai-")
    '\\\\bai-'
    """
    if not term:
        raise ValueError("empty term")
    escaped = "".join(("\\" + c) if c in _METACHARS else c for c in term)
    leading = r"\b" if term[0].isalnum() else ""
    trailing = r"\b" if term[-1].isalnum() else ""
    return f"{leading}{escaped}{trailing}"


def load_dictionary(path: Path | str) -> Dictionary:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    version = raw["dictionary_version"]
    families_raw = raw["families"]

    seen: set[tuple[str, str]] = set()
    terms_list: list[Term] = []
    family_patterns_acc: dict[str, list[str]] = {}

    for family_key, fcfg in families_raw.items():
        display = fcfg["display_name"]  # noqa: F841 — validated by direct access
        for entry in fcfg["terms"]:
            if isinstance(entry, str):
                canonical = entry
                override: str | None = None
            else:
                canonical = entry["canonical"]
                override = entry.get("regex_override")
            canonical_lc = canonical.lower()
            key = (family_key, canonical_lc)
            if key in seen:
                raise ValueError(
                    f"duplicate term in family {family_key!r}: {canonical!r}"
                )
            seen.add(key)
            pattern = override if override is not None else term_to_pattern(canonical_lc)
            terms_list.append(
                Term(canonical=canonical_lc, family=family_key, pattern=pattern)
            )
            family_patterns_acc.setdefault(family_key, []).append(pattern)

    family_patterns = {
        fk: "(?:" + "|".join(pats) + ")" for fk, pats in family_patterns_acc.items()
    }
    union_pattern = "(?:" + "|".join(t.pattern for t in terms_list) + ")"

    return Dictionary(
        version=version,
        families={k: v["display_name"] for k, v in families_raw.items()},
        terms=tuple(terms_list),
        family_patterns=family_patterns,
        union_pattern=union_pattern,
    )


def scan(text: str, d: Dictionary) -> dict[str, bool]:
    """Per-family hit flags for one document. Used by tests and audits;
    the bulk sweep happens in DuckDB."""
    lo = text.lower()
    return {fam: bool(re.search(pat, lo)) for fam, pat in d.family_patterns.items()}


def matched_terms(text: str, d: Dictionary) -> list[tuple[str, str]]:
    """Return (family, canonical) for every term that matched at least
    once, in declaration order."""
    lo = text.lower()
    return [(t.family, t.canonical) for t in d.terms if re.search(t.pattern, lo)]


def count_total_matches(text: str, d: Dictionary) -> int:
    """Total term-occurrences in `text` (intensity). Equivalent to
    DuckDB len(regexp_extract_all(text, union_pattern))."""
    return len(re.findall(d.union_pattern, text.lower()))

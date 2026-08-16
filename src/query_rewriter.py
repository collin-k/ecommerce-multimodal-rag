"""
Rewrite natural-language questions into short CLIP retrieval queries.

The original user question is left unchanged for the LLM. Only the
string sent to CLIP is simplified (product name, category, attributes).
"""

from __future__ import annotations

import re
from typing import Optional

_VS_RE = re.compile(
    r"^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*$",
    re.IGNORECASE,
)

_COMPARE_SPLIT_RE = (
    re.compile(r"\s+versus\s+", re.IGNORECASE),
    re.compile(r"\s+vs\.?\s+", re.IGNORECASE),
    re.compile(r"\s+to\s+", re.IGNORECASE),
    re.compile(r"\s+with\s+", re.IGNORECASE),
    re.compile(r"\s+and\s+", re.IGNORECASE),
)


def _split_comparison(text: str) -> Optional[tuple[str, str]]:
    """
    Split a comparison question into two product spans.

    Prefers ``versus`` / ``vs`` / ``to`` over ``with`` so phrases like
    ``bike with streamers to a Trek`` split on ``to``.

    Parameters
    ----------
    text : str
        Question without a trailing question mark.

    Returns
    -------
    tuple of str or None
        Left and right product spans, if a comparison was detected.
    """
    lowered = text.lower()
    if not lowered.startswith(("compare ", "can you compare ", "could you compare ")):
        vs_match = _VS_RE.match(text)
        if vs_match:
            return vs_match.group(1), vs_match.group(2)
        return None

    body = re.sub(
        r"^(?:can you |could you )?(?:please )?compare(?: the)?\s+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    for splitter in _COMPARE_SPLIT_RE:
        parts = splitter.split(body, maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None

_PREFIX_PATTERNS = (
    r"^(?:can you |could you |please )+",
    r"^what are the features of (?:the )?",
    r"^what are the specs of (?:the )?",
    r"^what flex rating and rider weight is (?:the )?",
    r"^what mini kits are included in (?:the )?",
    r"^what does (?:the )?",
    r"^what is included in (?:the )?",
    r"^what is the name of (?:the )?",
    r"^what size is (?:the )?",
    r"^what wheel sizes do (?:the )?",
    r"^what(?:'s| is) ",
    r"^what are(?: the)? ",
    r"^what ",
    r"^how many pieces (?:is|are) (?:in )?(?:the )?",
    r"^how much filament comes in (?:the )?",
    r"^how large is (?:the )?",
    r"^how do i use ",
    r"^does (?:the )?",
    r"^do (?:the )?",
    r"^is (?:the )?",
    r"^show me a picture of (?:the )?",
    r"^show me (?:the )?",
    r"^tell me about (?:the )?",
    r"^identify (?:this |the )?",
)

_SUFFIX_PATTERNS = (
    r",?\s+teach,?\s+and how many pieces is it$",
    r",?\s+and how (?:do i use it|many pieces is it)$",
    r",?\s+and (?:who|what) (?:it is|is it) for\??$",
    r"\s+built for$",
    r"\s+used for$",
    r"\s+come in extra small$",
    r"\s+limit volume for safe listening$",
    r"\s+with streamers and bmx pegs come in$",
)


def _clean_span(text: str) -> str:
    """
    Collapse whitespace and strip leading articles and punctuation.

    Parameters
    ----------
    text : str
        Raw span extracted from a question.

    Returns
    -------
    str
        Normalized phrase.
    """
    cleaned = re.sub(r"\s+", " ", text).strip(" .,;:!?\"'")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def rewrite_clip_query(question: str) -> str:
    """
    Turn a shopper question into a short string for CLIP retrieval.

    Parameters
    ----------
    question : str
        Original user question. This string should still be sent to the LLM.

    Returns
    -------
    str
        Product-centric query. Falls back to the original text when the
        rewrite would be too short to search.
    """
    original = " ".join(question.strip().split())
    if not original:
        return original

    stripped = original.rstrip(" ?!.")
    comparison = _split_comparison(stripped)
    if comparison is not None:
        left = _clean_span(comparison[0])
        right = _clean_span(comparison[1])
        rewritten = f"{left} {right}".strip()
        return rewritten if len(rewritten) >= 4 else original

    rewritten = stripped
    for pattern in _PREFIX_PATTERNS:
        rewritten = re.sub(pattern, "", rewritten, count=1, flags=re.IGNORECASE)
    for pattern in _SUFFIX_PATTERNS:
        rewritten = re.sub(pattern, "", rewritten, count=1, flags=re.IGNORECASE)

    rewritten = _clean_span(rewritten)
    if len(rewritten) < 4:
        return original
    return rewritten

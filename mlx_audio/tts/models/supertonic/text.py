"""Text preprocessing and unicode-indexed tokenization for Supertonic-2."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Union
from unicodedata import normalize as _unicode_normalize

import numpy as np

SUPPORTED_LANGUAGES = ("en", "ko", "es", "pt", "fr")

# Sentence-split separators that preserve the terminator.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+")
_TRAILING_PUNCT = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")

DEFAULT_CHUNK_MAX_LEN = {
    "en": 300,
    "es": 300,
    "pt": 300,
    "fr": 300,
    "ko": 120,
}


def preprocess_text(text: str, lang: str) -> str:
    """NFKD-normalize, collapse whitespace, ensure trailing punctuation, and
    wrap in ``<{lang}>…</{lang}>`` for the unicode indexer."""
    t = _unicode_normalize("NFKD", text)
    t = re.sub(r"\s+", " ", t).strip()
    if not _TRAILING_PUNCT.search(t):
        t += "."
    return f"<{lang}>{t}</{lang}>"


def encode_text(text: str, indexer: Union[Dict, Sequence[int]]) -> np.ndarray:
    """Map unicode codepoints to token ids using the supplied indexer.

    The indexer is the raw ``unicode_indexer.json`` content — either a list
    long enough to index directly by codepoint, or a dict whose keys are
    stringified codepoints.
    """
    ids: List[int] = []
    if isinstance(indexer, dict):
        for c in text:
            key = str(ord(c))
            if key in indexer:
                ids.append(int(indexer[key]))
            elif ord(c) in indexer:
                ids.append(int(indexer[ord(c)]))
            else:
                raise KeyError(f"codepoint {ord(c)!r} ({c!r}) not in unicode_indexer")
    else:
        for c in text:
            cp = ord(c)
            if cp >= len(indexer):
                raise KeyError(f"codepoint {cp} out of range for unicode_indexer")
            ids.append(int(indexer[cp]))
    return np.array([ids], dtype=np.int64)


def chunk_text(text: str, max_len: int) -> List[str]:
    """Split text into chunks no longer than ``max_len`` chars, at sentence
    boundaries when possible."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def resolve_chunk_max_len(lang: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return DEFAULT_CHUNK_MAX_LEN.get(lang, 300)

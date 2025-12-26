"""Text preprocessing utilities.

These functions intentionally mirror the preprocessing steps in the notebook:
- lowercasing
- URL/user/hashtag cleanup
- basic punctuation/digit cleanup
- whitespace normalization
- lemmatization (WordNet)

You can adjust these steps, but keep changes version-controlled to preserve reproducibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

import nltk
from nltk.stem import WordNetLemmatizer


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessConfig:
    remove_hashtags: bool = True
    remove_mentions: bool = True
    remove_urls: bool = True
    remove_non_alpha: bool = True
    lowercase: bool = True


_lemmatizer = WordNetLemmatizer()


def ensure_nltk() -> None:
    """Ensure required NLTK resources are available."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def clean_text(text: str, cfg: PreprocessConfig | None = None) -> str:
    """Clean and lemmatize a single text string."""
    if cfg is None:
        cfg = PreprocessConfig()

    if text is None:
        return ""

    s = str(text)

    if cfg.lowercase:
        s = s.lower()

    if cfg.remove_urls:
        s = _URL_RE.sub(" ", s)

    if cfg.remove_mentions:
        s = _USER_RE.sub(" ", s)

    if cfg.remove_hashtags:
        s = _HASHTAG_RE.sub(" ", s)

    if cfg.remove_non_alpha:
        s = _NON_ALPHA_RE.sub(" ", s)

    s = _WS_RE.sub(" ", s).strip()

    # Lemmatize token-wise
    ensure_nltk()
    tokens = [_lemmatizer.lemmatize(tok) for tok in s.split()]
    return " ".join(tokens)


def batch_clean(texts: Iterable[str], cfg: PreprocessConfig | None = None) -> List[str]:
    """Clean a sequence of texts."""
    return [clean_text(t, cfg) for t in texts]

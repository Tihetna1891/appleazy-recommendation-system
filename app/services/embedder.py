# app/services/embedder.py
"""
Embedding helpers (OpenAI)

• Centralises all model / batch / truncation settings via app.config
• Raises RuntimeError on any embedding failure (no silent empty lists)
• Supports single-string or list-of-strings input
"""

from __future__ import annotations

import os
from typing import Iterable, List

import openai
from dotenv import load_dotenv

from config import settings

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

load_dotenv()  # pull OPENAI_API_KEY from .env if present

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

EMBED_MODEL      = settings.EMBED_MODEL          # e.g. "text-embedding-3-small"
MAX_BATCH_SIZE   = settings.EMBED_BATCH_SIZE     # e.g. 96
MAX_CHARS        = settings.EMBED_MAX_CHARS      # e.g. 6000

# ------------------------------------------------------------------ #
#  Low-level helpers
# ------------------------------------------------------------------ #

def _chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    """Yield successive size-chunked lists."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _embed_batch(batch: List[str]) -> List[List[float]]:
    """
    Embed one batch; raise on any API error.
    Returns list[list[float]] (same order as input).
    """
    response = openai.Embeddings.create(model=EMBED_MODEL, input=batch)
    return [item.embedding for item in response.data]


def _embed_raw(texts: List[str]) -> List[List[float]]:
    """Embed the entire list, respecting MAX_BATCH_SIZE & MAX_CHARS."""
    all_vectors: List[List[float]] = []
    for batch in _chunked(texts, MAX_BATCH_SIZE):
        clipped = [t[:MAX_CHARS] for t in batch]
        try:
            all_vectors.extend(_embed_batch(clipped))
        except Exception as exc:
            raise RuntimeError(f"[embedder] OpenAI embedding failed → {exc}") from exc
    return all_vectors

# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def embed(texts: str | List[str]) -> List[float] | List[List[float]]:
    """
    Embed a single string or list of strings.
      • str  → single vector (list[float])
      • list → list of vectors
    """
    if isinstance(texts, str):
        vecs = _embed_raw([texts])
        if not vecs:
            raise RuntimeError("OpenAI returned no vector for single input")
        return vecs[0]
    return _embed_raw(texts)


# ------------------------------------------------------------------ #
#  Domain-specific helpers
# ------------------------------------------------------------------ #

def embed_user(user: dict) -> List[float]:
    """Combine résumé/CV text (if any) and embed into one vector."""
    resume_text = (user.get("resume") or "").strip()
    cv_text     = (user.get("cv") or "").strip()

    if resume_text and cv_text:
        combined = f"{cv_text}\n\n{resume_text}"
    elif cv_text:
        combined = cv_text
    elif resume_text:
        combined = resume_text
    else:
        combined = ", ".join(
            [
                user.get("position", ""),
                user.get("preferredIndustry", ""),
                user.get("ethnicity", ""),
                user.get("location", ""),
            ]
        ).strip() or "empty profile"

    return embed(combined)  # type: ignore[return-value]


def embed_jobs(job_list: List[dict]) -> List[List[float]]:
    """Embed each job’s description (or title+company fallback)."""
    texts: List[str] = []
    for job in job_list:
        if job.get("description"):
            texts.append(job["description"])
        else:
            texts.append(f"{job.get('jobTitle', '')} {job.get('company', '')}")
    return embed(texts)  # list[list[float]]

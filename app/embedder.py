from __future__ import annotations

import os
from typing import List

import openai
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")

EMBED_MODEL = "text-embedding-3-small"
MAX_BATCH_SIZE = 96
MAX_CHARS = 6000  # truncate long strings to avoid token overflow

# --------------------------------------------------------------------------- #
#  OpenAI embedding helpers
# --------------------------------------------------------------------------- #

def _chunked(lst: List[str], size: int) -> List[List[str]]:
    """Split list into chunks of given size."""
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def _embed_raw(texts: List[str]) -> List[List[float]]:
    embeddings = []
    try:
        for batch in _chunked(texts, MAX_BATCH_SIZE):
            clean_batch = [t[:MAX_CHARS] for t in batch]  # truncate overly long text
            response = openai.embeddings.create(model=EMBED_MODEL, input=clean_batch)
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"[embedder] Embedding failed: {e}")
    return embeddings

def embed(texts: List[str] | str) -> List[List[float]] | List[float]:
    if isinstance(texts, str):
        return _embed_raw([texts])[0]  # type: ignore
    return _embed_raw(texts)

# --------------------------------------------------------------------------- #
#  Convenience functions
# --------------------------------------------------------------------------- #

def embed_user(user: dict) -> List[float]:
    resume_text = (user.get("resume") or "").strip()
    cv_text = (user.get("cv") or "").strip()

    combined_text = ""
    if cv_text and resume_text:
        combined_text = f"{cv_text}\n\n{resume_text}"
    elif cv_text:
        combined_text = cv_text
    elif resume_text:
        combined_text = resume_text
    else:
        combined_text = ", ".join([
            user.get("position", ""),
            user.get("preferredIndustry", ""),
            user.get("ethnicity", ""),
            user.get("location", "")
        ]).strip() or "empty profile"

    return embed(combined_text)  # type: ignore

def embed_jobs(job_list: List[dict]) -> List[List[float]]:
    texts = []
    for job in job_list:
        if job.get("description"):
            texts.append(job["description"])
        else:
            texts.append(f"{job.get('jobTitle', '')} {job.get('company', '')}")
    return embed(texts)  # returns list[list[float]]

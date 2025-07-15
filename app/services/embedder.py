from __future__ import annotations
import os
from typing import Iterable, List
import openai
import tempfile
import pdfplumber
import requests
from dotenv import load_dotenv
from app.config import settings

# ------------------------------------------------------------------ #
#  Configuration
# ------------------------------------------------------------------ #

load_dotenv()  

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

EMBED_MODEL      = settings.EMBED_MODEL          
MAX_BATCH_SIZE   = settings.EMBED_BATCH_SIZE     
MAX_CHARS        = settings.EMBED_MAX_CHARS      

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
    response = openai.embeddings.create(model=EMBED_MODEL, input=batch)
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

def _download_pdf(url: str) -> str | None:
    """Download a remote PDF → return temp‑file path or None on failure."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as exc:
        print(f"[embedder] PDF download failed: {exc}")
        return None

    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(r.content)
    return path

def extract_text_from_pdf(url: str) -> str:
    """Return concatenated text of all pages or empty string on failure."""
    pdf_path = _download_pdf(url)
    if not pdf_path:
        return ""

    full_text: List[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                full_text.append(page.extract_text() or "")
    except Exception as exc:
        print(f"[embedder] PDF parsing failed: {exc}")
    finally:
        os.remove(pdf_path)

    return "\n".join(full_text).strip()

# ------------------------------------------------------------------ #
#  Main Function
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
    resume_field = (user.get("resume") or "").strip()
    cv_field = (user.get("cv") or "").strip()

    content = ""

    if resume_field:
        if resume_field.lower().endswith(".pdf"):
            content = extract_text_from_pdf(resume_field).strip()
        else:
            content = resume_field
    elif cv_field:
        content = cv_field

    if not content:
        content = ", ".join([
            user.get("position", ""),
            user.get("preferredIndustry", ""),
            user.get("ethnicity", ""),
            user.get("location", ""),
        ]).strip() or "empty profile"

    return embed(content)  



def embed_jobs(job_list: List[dict]) -> List[List[float]]:
    """Embed each job’s description (or title+company fallback)."""
    texts: List[str] = []
    for job in job_list:
        if job.get("description"):
            texts.append(job["description"])
        else:
            texts.append(f"{job.get('jobTitle', '')} {job.get('company', '')}")
    return embed(texts)  # list[list[float]]

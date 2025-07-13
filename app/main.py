from fastapi import FastAPI, HTTPException, Query
from typing import List
import numpy as np
import requests

from embedder import embed_user, embed_jobs
from vector_calculate import build_job_matrix, rank_jobs_for_user
from redis_client import (
    get_recommendations,
    store_recommendations,
    get_job_by_id,
    store_jobs,
)

app = FastAPI()

USERS_API = "https://sandbox.appleazy.com/api/v1/user/profiles"
JOBS_API = "https://server.appleazy.com/api/v1/job-listing"


# ────────────────────────────────
# Helper: Fetch user & jobs
# ────────────────────────────────
def fetch_user(user_id: str) -> dict | None:
    r = requests.get(f"{USERS_API}/{user_id}")
    return r.json() if r.status_code == 200 else None


def fetch_jobs_open(limit: int = 1000) -> list[dict]:
    r = requests.get(JOBS_API, params={"status": "open", "limit": limit})
    return r.json().get("jobs", []) if r.status_code == 200 else []


# ────────────────────────────────
# API: Get Recommendations
# ────────────────────────────────
@app.get("/recommendations/{user_id}")
def recommendations(
    user_id: str,
    page: int = Query(1, gt=0),
    size: int = Query(1000, gt=0, le=1000),  # allow full fetch
):
    print(f"🔎 Getting recommendations for user: {user_id}")

    # 1️⃣ Try Redis cache
    cached = get_recommendations(user_id)
    if cached:
        print(f"⚡️ Cache hit for user {user_id}. Returning {len(cached)} cached job-score pairs.")
        return _paged_response(cached, page, size)

    # 2️⃣ No cache → compute
    user = fetch_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")

    user_vec = np.asarray(embed_user(user), dtype=np.float32)

    jobs_open = fetch_jobs_open()
    if not jobs_open:
        raise HTTPException(500, "No open jobs available")

    job_vecs = embed_jobs(jobs_open)
    store_jobs(jobs_open, job_vecs)

    job_ids = [j["id"] for j in jobs_open]
    job_matrix = build_job_matrix(job_vecs)
    ranked = rank_jobs_for_user(user_vec, job_matrix)  # [(idx, score)]
    ranked_pairs = [[job_ids[i], float(score)] for i, score in ranked]

    store_recommendations(user_id, ranked_pairs)

    return _paged_response(ranked_pairs, page, size)


# ────────────────────────────────
# Internal: paginate + hydrate
# ────────────────────────────────
def _paged_response(
    ranked_pairs: List[list],  # list of [job_id, score]
    page: int,
    size: int,
):
    print(f"📄 Paginating results → Page {page}, Size {size}")

    hydrated: List[dict] = []
    for pair in ranked_pairs:
        if isinstance(pair, list) and len(pair) == 2:
            job_id, score = pair
        else:
            job_id, score = pair, None

        job = get_job_by_id(job_id)
        if job:
            job["score"] = score
            print(f"✅ Fetched {job_id} from Redis (score: {score:.4f})")
            hydrated.append(job)
        else:
            print(f"⚠️ Job ID {job_id} not found in Redis.")

    start = (page - 1) * size
    end = start + size
    return {
        "total": len(hydrated),
        "page": page,
        "size": size,
        "jobs": hydrated[start:end],
    }

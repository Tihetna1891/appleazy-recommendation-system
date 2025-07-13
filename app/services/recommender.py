# services/recommender.py
from fastapi import HTTPException
import numpy as np
from typing import List

from services.redis_client import (
    redis_client,
    get_recommendations,
    store_recommendations,
    get_job_by_id,
    store_jobs,
)
from services.vector_calculate import build_job_matrix, rank_jobs_for_user
from services.embedder import embed_user, embed_jobs
from services.fetcher import fetch_user, fetch_jobs_open


def get_user_recommendations(user_id: str, page: int, size: int):
    print(f"🔎 Getting recommendations for user: {user_id}")

    cached = get_recommendations(user_id)
    if cached:
        print(f"⚡️ Cache hit for user {user_id}. Returning {len(cached)} cached job IDs.")
        return _paged_response(cached, page, size)

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
    ranked = rank_jobs_for_user(user_vec, job_matrix)
    ranked_ids = [[job_ids[i], float(score)] for i, score in ranked]

    store_recommendations(user_id, ranked_ids)
    return _paged_response(ranked_ids, page, size)

def get_all_recommendations():
    pattern = "recommendations:*"
    keys = redis_client.keys(pattern)
    
    results = {}
    for key in keys:
        user_id = key.decode().split(":")[1]  # Extract user_id from key
        recs = redis_client.get(key)
        if recs:
            results[user_id] = json.loads(recs)
    
    return results

def _paged_response(ranked_ids: List[list], page: int, size: int):
    hydrated = []
    for job_id, score in ranked_ids:
        job = get_job_by_id(job_id)
        if job:
            job["score"] = score
            hydrated.append(job)

    start = (page - 1) * size
    end = start + size
    print(f"📦 Returning {len(hydrated)} jobs out of {len(ranked_ids)} total matched.")
    return {
        "total": len(hydrated),
        "page": page,
        "size": size,
        "jobs": hydrated[start:end],
    }

def get_all_recommendations():
    result = []
    keys = redis_client.keys("reco:*")

    for key in keys:
        user_id = key.decode().split(":")[1]
        job_scores = get_recommendations(user_id)
        if not job_scores:
            continue

        jobs = [
            {**get_job_by_id(job_id), "score": score}
            for job_id, score in job_scores
            if get_job_by_id(job_id) is not None
        ]

        result.append({
            "user_id": user_id,
            "total": len(jobs),
            "jobs": jobs,
        })

    return result
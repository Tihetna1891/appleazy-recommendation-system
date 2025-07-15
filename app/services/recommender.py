# services/recommender.py
from fastapi import HTTPException
import numpy as np
from typing import List
import requests
import json

from services.redis_client import (
    redis_client,
    get_recommendations,
    store_recommendations,
    get_job_by_id,
    store_jobs, store_user_vector, get_all_jobs
)
from services.vector_calculate import build_job_matrix, rank_jobs_for_user
from services.embedder import embed_user, embed_jobs
from services.fetcher import fetch_user, fetch_jobs_open


def get_user_recommendations(user_id: str, page: int, size: int):
    print(f" Getting recommendations for user: {user_id}")

    cached = get_recommendations(user_id)
    if cached:
        print(f" Cache hit for user {user_id}. Returning {len(cached)} cached job IDs.")
        return _paged_response(cached, page, size)

    # Fetch user
    user = fetch_user(user_id)
    if not user:
        print(f" User {user_id} not found in main DB. Trying fallback to generate recommendation.")
        try:
            generate_recommendation_for_user(user_id)
            cached = get_recommendations(user_id)
            if cached:
                return _paged_response(cached, page, size)
            else:
                raise HTTPException(500, f"Failed to generate recommendations for {user_id}")
        except Exception as e:
            print(f" Failed to generate recommendation: {e}")
            raise HTTPException(404, f"User {user_id} not found and could not generate recommendation")

    # If user exists but no cached recs (first-time call)
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

def generate_recommendation_for_user(user_id: str):
    #  1. Fetch the user
    user_resp = requests.get(f"https://sandbox.appleazy.com/api/v1/user/get-profile/{user_id}")
    if user_resp.status_code != 200:
        raise ValueError(f"User {user_id} not found")

    user_data = user_resp.json()
    user_vector = embed_user(user_data)
    store_user_vector(user_id, user_vector)

    #  2. Load existing jobs from Redis
    jobs = get_all_jobs()  # make sure you have a function for this
    if not jobs:
        raise ValueError("No jobs in the system")

    job_ids = [job["id"] for job in jobs]
    job_vectors = [job["vector"] for job in jobs]
    job_matrix = build_job_matrix(job_vectors)

    #  3. Rank
    scored = rank_jobs_for_user(np.array(user_vector, dtype=np.float32), job_matrix)
    ranked_pairs = [[job_ids[i], float(score)] for i, score in scored]

    #  4. Store recommendations
    store_recommendations(user_id, ranked_pairs)
    print(f" Recommendations created for user {user_id}")
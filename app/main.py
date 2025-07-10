from fastapi import FastAPI, HTTPException, Query
from typing import List
import numpy as np
import requests
from redis_client import get_recommendations, store_recommendations
from vector_calculate import calculate_similarity
from scheduler import start_scheduler

app = FastAPI()

USERS_API = "https://sandbox.appleazy.com/api/v1/user/profiles"
JOBS_API = "https://server.appleazy.com/api/v1/job-listing"

def fetch_user_vector(user_id: int):
    response = requests.get(f"{USERS_API}/{user_id}")
    if response.status_code == 200:
        return np.array(response.json()["vector"])
    return None

def fetch_all_jobs():
    response = requests.get(JOBS_API)
    if response.status_code == 200:
        jobs = response.json()
        return {job["id"]: np.array(job["vector"]) for job in jobs}
    return {}

@app.on_event("startup")
def startup_event():
    start_scheduler(fetch_all_users, fetch_all_jobs)

def fetch_all_users():
    response = requests.get(USERS_API)
    if response.status_code == 200:
        users = response.json()
        return {user["id"]: np.array(user["vector"]) for user in users}
    return {}

@app.get("/recommendations/{user_id}")
def get_user_recommendations(user_id: int, page: int = Query(1), size: int = Query(20)):
    cached = get_recommendations(user_id)
    if cached:
        start = (page - 1) * size
        return {"jobs": cached[start:start + size]}

    user_vector = fetch_user_vector(user_id)
    if user_vector is None:
        raise HTTPException(status_code=404, detail="User not found")

    jobs = fetch_all_jobs()
    if not jobs:
        raise HTTPException(status_code=500, detail="Could not fetch jobs")

    job_vectors = list(jobs.values())
    job_ids = list(jobs.keys())

    scored = calculate_similarity(user_vector, job_vectors)
    sorted_job_ids = [job_ids[i] for i, _ in scored]

    store_recommendations(user_id, sorted_job_ids)
    start = (page - 1) * size
    return {"jobs": sorted_job_ids[start:start + size]}

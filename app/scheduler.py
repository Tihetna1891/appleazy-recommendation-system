from apscheduler.schedulers.background import BackgroundScheduler
from redis_client import (
    get_user_vector,
    store_jobs,
    store_recommendations,
    get_recommendations,
)
from vector_calculate import calculate_similarity
from embedder import embed_jobs
import numpy as np
import requests

USERS_API = "https://sandbox.appleazy.com/api/v1/user/profiles"
JOBS_API  = "https://server.appleazy.com/api/v1/job-listing"

def fetch_all_users():
    r = requests.get(USERS_API, params={"limit": 1000})
    return r.json().get("users", []) if r.status_code == 200 else []

def fetch_all_jobs():
    r = requests.get(JOBS_API, params={"status": "open", "limit": 1000})
    return r.json().get("jobs", []) if r.status_code == 200 else []

def run_daily_recalculation():
    print("⏰ [Scheduler] Running daily job-user similarity update...")

    # Fetch and store jobs
    jobs = fetch_all_jobs()
    if not jobs:
        print("⚠️ No jobs fetched.")
        return
    job_ids = [j["id"] for j in jobs]
    job_vecs = embed_jobs(jobs)
    store_jobs(jobs, job_vecs)

    # Fetch user IDs and recompute matches
    users = fetch_all_users()
    if not users:
        print("⚠️ No users fetched.")
        return

    for user in users:
        user_id = user.get("id")
        if not user_id:
            continue

        user_vec = get_user_vector(user_id)
        if user_vec is None:
            print(f"⛔️ User vector missing for {user_id}, skipping.")
            continue

        # Recompute similarity with updated jobs
        scored = calculate_similarity(np.array(user_vec, dtype=np.float32), job_vecs)
        ranked_pairs = [[job_ids[i], float(score)] for i, score in scored]
        store_recommendations(user_id, ranked_pairs)
        print(f"✅ Updated recommendations for user {user_id}")

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Africa/Addis_Ababa")
    scheduler.add_job(
        run_daily_recalculation,
        trigger='cron',
        hour=0,
        minute=0,
        id="daily_job_user_update"
    )
    scheduler.start()

# services/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from app.services.fetcher import fetch_all_users, fetch_jobs_open
from app.services.redis_client import get_user_vector, store_jobs, store_recommendations, get_all_jobs, get_all_users
from app.services.embedder import embed_jobs
from app.services.vector_calculate import build_job_matrix, rank_jobs_for_user
import numpy as np
import logging

def run_daily_recalculation():
    print(" [Scheduler] Running daily job-user similarity update...")

    jobs = fetch_jobs_open()

    if not jobs:
        print(" No new jobs fetched from API. Using existing jobs in Redis.")
        jobs = get_all_jobs()

    if not jobs:
        print(" No jobs available in Redis. Aborting.")
        return

    job_ids = [j["id"] for j in jobs]
    job_vecs = embed_jobs(jobs)
    job_matrix = build_job_matrix(job_vecs)
    store_jobs(jobs, job_vecs)

    users = fetch_all_users()
    if not users:
        print(" No users fetched from api.")
        users = get_all_users()
    print(f" Found {len(users)} users to process.")

    for user in users:
        user_id = user.get("userId")
        if not user_id:
            continue

        user_vec = get_user_vector(user_id)
        if user_vec is None:
            continue

        ranked_pairs = rank_jobs_for_user(np.array(user_vec, dtype=np.float32), job_matrix)
        # Convert job_idx to job_id in ranked pairs
        job_scores = [[job_ids[i], score] for i, score in ranked_pairs]
        store_recommendations(user_id, job_scores)
        print(f" Updated recommendations for user {user_id}")


def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Africa/Addis_Ababa")
    logger = logging.getLogger("apscheduler")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.info("Initial run to populate recommendations...")
    # Initial run to populate recommendations
    scheduler.add_job(run_daily_recalculation, id="initial_calculation")
    scheduler.add_job(run_daily_recalculation, trigger='cron', hour=22, minute=0, id="daily_job_user_update")
    scheduler.start()

from apscheduler.schedulers.background import BackgroundScheduler
from redis_client import store_recommendations, get_recommendations
from vector_score import calculate_similarity
import numpy as np
from typing import Dict

def update_user_scores_if_missing(users: Dict[int, np.ndarray], jobs: Dict[int, np.ndarray]):
    for user_id, user_vec in users.items():
        # Check if already in Redis
        if get_recommendations(user_id) is not None:
            continue
        
        job_vectors = list(jobs.values())
        job_ids = list(jobs.keys())
        scored = calculate_similarity(user_vec, job_vectors)
        sorted_ids = [job_ids[i] for i, _ in scored]
        store_recommendations(user_id, sorted_ids)
        print(f"Stored recommendations for user {user_id}")

def start_scheduler(users: Dict[int, np.ndarray], jobs: Dict[int, np.ndarray]):
    scheduler = BackgroundScheduler(timezone="Africa/Addis_Ababa")  # Or UTC if preferred

    scheduler.add_job(
        lambda: update_user_scores_if_missing(users, jobs),
        trigger='cron',
        hour=0,
        minute=0,
        id="daily_score_update"
    )

    scheduler.start()

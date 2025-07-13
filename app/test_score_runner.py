from time import perf_counter

from test_data import users, jobs
from embedder import embed_user, embed_jobs
from vector_calculate import build_job_matrix, rank_jobs_for_user
from redis_client import (
    store_recommendations,
    store_user_vector,
    store_jobs,
)

import numpy as np


def run_manual_scoring():
    print("⚡ Running fast job-user scoring with Redis store …\n")

    # -------- BATCH-EMBED + NORMALIZE THE JOBS  -----------------
    job_ids  = list(jobs.keys())
    job_list = [jobs[jid] for jid in job_ids]
    
    t0 = perf_counter()
    job_vecs = embed_jobs(job_list)
    job_matrix = build_job_matrix(job_vecs)
    print(f"✅ Embedded and normalized {len(job_vecs)} jobs in {perf_counter()-t0:.2f}s\n")

    # -------- SCORE PER USER + STORE IN REDIS -------------------
    for user_id, user_data in users.items():
        if not user_data:
            print(f"⚠️ Skipping empty user: {user_id}")
            continue

        try:
            user_vec = np.array(embed_user(user_data))
        except Exception as e:
            print(f"❌ Failed to embed user {user_id}: {e}")
            continue

        scored = rank_jobs_for_user(user_vec, job_matrix)
        sorted_results = [(job_ids[i], score) for i, score in scored]

        store_recommendations(user_id, sorted_results)

        print(f"👤  User: {user_id} — All Job Matches (sorted):")
        for rank, (job_id, score) in enumerate(sorted_results, 1):
            j = jobs[job_id]
            print(f"  {rank}. {j['jobTitle']} at {j['company']} "
                  f"({j['location']})  →  {score:.4f}")
        print("=" * 60)



if __name__ == "__main__":
    run_manual_scoring()

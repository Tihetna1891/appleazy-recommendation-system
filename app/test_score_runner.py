from test_data import users, jobs
from embedder import embed_user, embed
from vector_calculate import calculate_similarity

def run_manual_scoring():
    print("🔍 Running manual vector scoring...\n")

    # Convert job descriptions to vectors
    job_ids = list(jobs.keys())
    job_vectors = [embed(jobs[job_id]["description"]) for job_id in job_ids]

    for user_id, user_data in users.items():
        user_vector = embed_user(user_data)

        # Calculate cosine similarity
        scored = calculate_similarity(user_vector, job_vectors)  # List of (job_index, score)
        sorted_results = [(job_ids[i], score) for i, score in scored]

        print(f"\n👤 User: {user_id}")
        print("Top Job Matches:")
        for rank, (job_id, score) in enumerate(sorted_results[:5], 1):
            job = jobs[job_id]
            print(f"  {rank}. {job['jobTitle']} at {job['company']} ({job['location']})")
            print(f"     → Similarity Score: {score:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    run_manual_scoring()

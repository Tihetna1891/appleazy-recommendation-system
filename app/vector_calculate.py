from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(user_vector, job_vectors):
    """
    Returns similarity scores between a user and all jobs
    """
    similarities = cosine_similarity([user_vector], job_vectors)[0]
    job_scores = list(enumerate(similarities))  # (job_index, score)
    job_scores.sort(key=lambda x: x[1], reverse=True)
    return job_scores

import numpy as np
from numpy.linalg import norm
from typing import Union

def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Return an L2-normalised copy of a 2-D array."""
    return mat / (norm(mat, axis=1, keepdims=True) + 1e-12)

# PRE-compute this once per job pool and keep it in memory!
def build_job_matrix(job_vectors) -> np.ndarray:
    """
    Convert a list/array of job vectors → 2-D normalised NumPy matrix.
    Call this ONCE after you embed all jobs, then reuse.
    """
    job_mat = np.asarray(job_vectors, dtype=np.float32)
    return _l2_normalise(job_mat)

def cosine_scores(
    user_vectors: Union[np.ndarray, list[list[float]]],
    job_mat_normalised: np.ndarray,
) -> np.ndarray:
    """
    Fast cosine similarity.
        user_vectors      – shape (d,) or (n_users, d)
        job_mat_normalised – shape (n_jobs, d)    (already L2-normalised!)

    Returns
        scores – shape (n_users, n_jobs)   (or (n_jobs,) if 1-D input)
    """
    user_mat = np.asarray(user_vectors, dtype=np.float32)
    if user_mat.ndim == 1:
        user_mat = user_mat[None, :]        # → shape (1, d)
    user_mat = _l2_normalise(user_mat)      # normalise once
    return user_mat @ job_mat_normalised.T  # matrix multiply gives cosine!

def rank_jobs_for_user(
    user_vector: np.ndarray,
    job_mat_normalised: np.ndarray,
) -> list[tuple[int, float]]:
    """
    Convenience wrapper for a single user.
    Returns list of (job_idx, score) sorted desc.
    """
    scores = cosine_scores(user_vector, job_mat_normalised)  # (1, n_jobs)
    idxs = np.argsort(-scores[0])                            # sort descending
    return [(int(i), float(scores[0, i])) for i in idxs]

# def rank_jobs_topk(
#     user_vector: np.ndarray,
#     job_mat_normalised: np.ndarray,
#     k: int = 20,
# ) -> list[tuple[int, float]]:
#     """Return only the top-k (job_idx, score) pairs."""
#     scores = cosine_scores(user_vector, job_mat_normalised)[0]          # (n_jobs,)
#     if k >= scores.size:
#         idxs = np.argsort(-scores)
#     else:
#         part = np.argpartition(-scores, k)[:k]          # O(n)
#         idxs = part[np.argsort(-scores[part])]          # final sort of k
#     return [(int(i), float(scores[i])) for i in idxs]


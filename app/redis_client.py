# redis_client.py
"""
Redis helpers

✓ Classic key-value cache for recommendations   (store_recommendations / get_recommendations)
✓ Vector storage + ANN search (HNSW) for job matching
"""

from __future__ import annotations

import json
import os
import struct
from typing import List
from datetime import timedelta
import redis

# --------------------------------------------------------------------------- #
#  Connection
# --------------------------------------------------------------------------- #

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=False,  # keep binary for vectors
)


# --------------------------------------------------------------------------- #
#  Simple recommendation cache (unchanged)
# --------------------------------------------------------------------------- #

def store_recommendations(user_id: str, sorted_job_ids: list[str], ttl: int = 86400):
    """Cache list of job-IDs for a user (1-hour TTL default)."""
    redis_client.set(f"reco:{user_id}", json.dumps(sorted_job_ids), ex=ttl)


def get_recommendations(user_id: str) -> list[str] | None:
    data = redis_client.get(f"reco:{user_id}")
    return json.loads(data) if data else None


# --------------------------------------------------------------------------- #
#  Vector search helpers
# --------------------------------------------------------------------------- #

VECTOR_DIM = 1536                # embedding size (text-embedding-3-small)
INDEX_NAME = "jobs_idx"
JOB_KEY_PREFIX = "job:"          # every job stored as  job:<id>

def _ser(vec: List[float]) -> bytes:
    """Serialize list[float] → binary (float32 little-endian)."""
    return struct.pack(f"{len(vec)}f", *vec)

# ---------- 1. Create / recreate the HNSW index ---------- #
def ensure_index():
    """
    Create the RedisSearch HNSW index if it doesn't exist.
    Safe to call multiple times.
    """
    from redis.commands.search.field import VectorField, TextField

    try:
        redis_client.ft(INDEX_NAME).info()
    except redis.ResponseError:
        # index doesn't exist → create
        redis_client.ft(INDEX_NAME).create_index([
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                    # HNSW tuning params (optional):
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
            TextField("jobTitle"),
            TextField("company"),
            TextField("location"),
        ])
        print("✅ Redis vector index created.")

# ---------- 2. Store (or upsert) job vectors ---------- #
def store_jobs(job_dicts: list[dict], vectors: list[list[float]]):
    """
    Bulk store job hashes with their vectors.
    `job_dicts` and `vectors` must be aligned by index.
    """
    pipe = redis_client.pipeline(transaction=False)
    for job, vec in zip(job_dicts, vectors):
        key = f"{JOB_KEY_PREFIX}{job['id']}"
        pipe.hset(
            key,
            mapping={
                "jobTitle": job.get("jobTitle", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "minPay": job.get("minPay", 0),
                "maxPay": job.get("maxPay", 0),
                "jobSetting": job.get("jobSetting", ""),
                "jobType": job.get("jobType", ""),
                "status": job.get("status", ""),
                "source": job.get("source", ""),
                "description": job.get("description", ""),
                "link": job.get("link", ""),
                "date": job.get("date", ""),
                "vector": _ser(vec),
            },
        )
        pipe.expire(key, timedelta(days=10))

    pipe.execute()
    print(f"✅ Stored/updated {len(job_dicts)} jobs in Redis.")


# ---------- 3. ANN query ---------- #
def knn_search(
    user_vec: list[float],
    k: int = 5,
    filters: str | None = None,
) -> list[tuple[str, float]]:
    """
    Return top-k (job_id, score) pairs most similar to `user_vec`.

    • `filters` allows RediSearch metadata filters, e.g. '@location:remote'
    """
    blob = _ser(user_vec)
    base_query = "*"
    if filters:
        base_query = f"({filters})"

    query = f"{base_query}=>[KNN {k} @vector $vec_param AS score]"
    res = redis_client.ft(INDEX_NAME).search(
        query,
        query_params={"vec_param": blob},
        sort_by="score",
        dialect=2,
    )
    return [(doc.id.split(":")[1], float(doc.score)) for doc in res.docs]

# ---------- 4. Store user vector ---------- #
def store_user_vector(user_id: str, vector: list[float]):
    """Store binary user vector to Redis as 'user:<id>'."""
    redis_client.set(f"user:{user_id}", _ser(vector))


# ---------- 5. Retrieve user vector ---------- #
def get_user_vector(user_id: str) -> list[float] | None:
    """Load and deserialize binary user vector from Redis."""
    raw = redis_client.get(f"user:{user_id}")
    if raw is None:
        return None
    return list(struct.unpack(f"{len(raw)//4}f", raw))

def get_job_by_id(job_id: str) -> dict | None:
    key = f"job:{job_id}"
    job_data = redis_client.hgetall(key)
    if not job_data:
        return None

    def decode(field): return job_data.get(field.encode(), b"").decode()

    return {
        "id": job_id,
        "jobTitle": decode("jobTitle"),
        "company": decode("company"),
        "location": decode("location"),
        "minPay": int(decode("minPay") or 0),
        "maxPay": int(decode("maxPay") or 0),
        "jobSetting": decode("jobSetting"),
        "jobType": decode("jobType"),
        "status": decode("status"),
        "source": decode("source"),
        "description": decode("description"),
        "link": decode("link"),
        "date": decode("date"),
        "metadata": {}  # Add if needed
    }



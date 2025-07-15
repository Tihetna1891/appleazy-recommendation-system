# app/services/redis_client.py
"""
Redis helpers:
✓ Classic key-value cache for recommendations
✓ Vector storage + ANN search (HNSW) for job matching
"""

from __future__ import annotations
import json
import struct
from typing import List
from datetime import timedelta

import redis
from redis.commands.search.field import VectorField, TextField

from app.config import settings

# --------------------------------------------------------------------------- #
#  Connection
# --------------------------------------------------------------------------- #

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=False,  # keep binary format for vectors
)

# --------------------------------------------------------------------------- #
#  Simple recommendation cache
# --------------------------------------------------------------------------- #

def store_recommendations(user_id: str, sorted_job_ids: list[str], ttl: int = 86400):
    """Cache list of job-IDs for a user with optional TTL (default: 1 day)."""
    redis_client.set(f"reco:{user_id}", json.dumps(sorted_job_ids), ex=ttl)


def get_recommendations(user_id: str) -> list[str] | None:
    """Retrieve cached recommendations for a user."""
    data = redis_client.get(f"reco:{user_id}")
    return json.loads(data) if data else None

# --------------------------------------------------------------------------- #
#  Vector indexing + storage
# --------------------------------------------------------------------------- #

def _ser(vec: List[float]) -> bytes:
    """Serialize list[float] → binary float32 LE."""
    return struct.pack(f"{len(vec)}f", *vec)


def ensure_index():
    """Ensure HNSW vector index exists (safe to call repeatedly)."""
    try:
        redis_client.ft(settings.REDIS_INDEX_NAME).info()
    except redis.ResponseError:
        redis_client.ft(settings.REDIS_INDEX_NAME).create_index([
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": settings.VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
            TextField("jobTitle"),
            TextField("company"),
            TextField("location"),
        ])
        print("✅ Redis vector index created.")


def store_jobs(job_dicts: list[dict], vectors: list[list[float]]):
    """Store job vectors + metadata in Redis. Keys auto-expire in 10 days."""
    pipe = redis_client.pipeline(transaction=False)
    for job, vec in zip(job_dicts, vectors):
        key = f"{settings.JOB_KEY_PREFIX}{job['id']}"
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
        pipe.expire(key, timedelta(seconds=settings.JOB_TTL_SECONDS))

    pipe.execute()
    print(f"✅ Stored/updated {len(job_dicts)} jobs in Redis.")


def knn_search(
    user_vec: list[float],
    k: int = 5,
    filters: str | None = None,
) -> list[tuple[str, float]]:
    """Return top-k (job_id, score) pairs most similar to `user_vec`."""
    blob = _ser(user_vec)
    query = f"({filters})=>[KNN {k} @vector $vec_param AS score]" if filters else f"*=>[KNN {k} @vector $vec_param AS score]"

    res = redis_client.ft(settings.REDIS_INDEX_NAME).search(
        query,
        query_params={"vec_param": blob},
        sort_by="score",
        dialect=2,
    )
    return [(doc.id.split(":")[1], float(doc.score)) for doc in res.docs]

# --------------------------------------------------------------------------- #
#  User vector storage
# --------------------------------------------------------------------------- #

def store_user_vector(user_id: str, vector: list[float]):
    """Store binary user vector to Redis."""
    redis_client.set(f"user:{user_id}", _ser(vector))


def get_user_vector(user_id: str) -> list[float] | None:
    """Load and deserialize binary user vector from Redis."""
    raw = redis_client.get(f"user:{user_id}")
    return list(struct.unpack(f"{len(raw)//4}f", raw)) if raw else None


# --------------------------------------------------------------------------- #
#  Job retrieval by ID
# --------------------------------------------------------------------------- #

def get_job_by_id(job_id: str) -> dict | None:
    """Retrieve full job metadata by ID."""
    key = f"{settings.JOB_KEY_PREFIX}{job_id}"
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
        "metadata": {}
    }

def get_all_jobs() -> list[dict]:
    keys = redis_client.keys("job:*")
    jobs = []
    for key in keys:
        job_id = key.decode().split(":")[1]
        job = get_job_by_id(job_id)
        if job:
            jobs.append({
                **job,
                "vector": get_job_vector_raw(job_id)  # you might need this helper
            })
    return jobs

def get_all_users() -> list[dict]:
    """Retrieve all stored users and their vectors from Redis."""
    keys = redis_client.keys("user:*")
    users = []
    for key in keys:
        user_id = key.decode().split(":")[1]
        vector = get_user_vector(user_id)
        if vector:
            users.append({
                "id": user_id,
                "vector": vector
            })
    return users


def get_job_vector_raw(job_id: str) -> list[float]:
    raw = redis_client.hget(f"job:{job_id}", "vector")
    if raw is None:
        return []
    return list(struct.unpack(f"{len(raw)//4}f", raw))

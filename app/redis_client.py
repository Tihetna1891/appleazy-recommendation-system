import redis
import json
import os

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

def store_recommendations(user_id, sorted_job_ids):
    redis_client.set(user_id, json.dumps(sorted_job_ids), ex=3600)  # 1-hour TTL

def get_recommendations(user_id):
    data = redis_client.get(user_id)
    return json.loads(data) if data else None

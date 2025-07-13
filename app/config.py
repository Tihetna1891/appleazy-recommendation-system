# app/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env into os.environ

class Settings:
    USERS_API: str = os.getenv("USERS_API", "https://sandbox.appleazy.com/api/v1/user/profiles")
    JOBS_API: str = os.getenv("JOBS_API", "https://server.appleazy.com/api/v1/job-listing")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    VECTOR_DIM: int = 1536
    JOB_TTL_SECONDS: int = 864000  # 10 days
    REDIS_INDEX_NAME = "jobs_idx"
    JOB_KEY_PREFIX = "job:"
    EMBED_MODEL       = "text-embedding-3-small"
    EMBED_BATCH_SIZE  = 96
    EMBED_MAX_CHARS   = 6000
settings = Settings()

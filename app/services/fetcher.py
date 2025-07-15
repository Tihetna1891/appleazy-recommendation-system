# app/services/fetcher.py

import requests
from app.config import settings


def fetch_user(user_id: str) -> dict | None:
    url = f"{settings.USERS_API}/{user_id}"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def fetch_all_users(limit: int = 1000) -> list[dict]:
    r = requests.get(settings.USERS_API, params={"limit": limit})
    return r.json().get("users", []) if r.status_code == 200 else []


def fetch_jobs_open(limit: int = 1000) -> list[dict]:
    params = {"status": "open", "limit": limit}
    r = requests.get(settings.JOBS_API, params=params)
    return r.json().get("jobs", []) if r.status_code == 200 else []



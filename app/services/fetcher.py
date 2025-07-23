# app/services/fetcher.py

import requests
from app.config import settings
from typing import Any


def fetch_user(user_id: str) -> dict | None:
    url = f"{settings.USERS_API}/{user_id}"
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


def fetch_all_users(limit: int | None = 100000) -> list[dict]:
    params: dict[str, Any] = {}
    if limit is not None:
        params["limit"] = limit
    r = requests.get(settings.USERS_API, params=params)
    return r.json().get("users", []) if r.status_code == 200 else []


def fetch_jobs_open(limit: int | None = 100000) -> list[dict]:
    params: dict[str, Any] = {"status": "open"}
    if limit is not None:
        params["limit"] = limit
    r = requests.get(settings.JOBS_API, params=params)
    return r.json().get("jobs", []) if r.status_code == 200 else []



import numpy as np
import pytest

from app.services import recommender

# ----------------- utilities we will monkey‑patch --------------------------- #
DUMMY_USER = {
    "id": "u1",
    "resume": "Skilled pythonista",
    "position": "Engineer",
    "preferredIndustry": "Tech",
    "ethnicity": "other",
    "location": "Remote",
}

DUMMY_JOBS = [
    {"id": "j1", "description": "Great job", "jobTitle": "Dev"},
    {"id": "j2", "description": "Another job", "jobTitle": "QA"},
]

def test_get_user_recommendations_pipeline(monkeypatch):
    monkeypatch.setattr(recommender, "get_job_by_id", lambda job_id: {"id": job_id, "title": "Mock Job"})
    monkeypatch.setattr(recommender, "fetch_user", lambda uid: DUMMY_USER)
    monkeypatch.setattr(recommender, "fetch_jobs_open", lambda: DUMMY_JOBS)
    monkeypatch.setattr(recommender, "embed_user", lambda u: [0.1, 0.2, 0.3])
    monkeypatch.setattr(recommender, "embed_jobs", lambda jobs: [[0.1, 0.1, 0.1]] * len(jobs))
    monkeypatch.setattr(recommender, "store_jobs", lambda *a, **k: None)
    monkeypatch.setattr(recommender, "store_recommendations", lambda *a, **k: None)
    monkeypatch.setattr(recommender, "get_recommendations", lambda uid: None)

    out = recommender.get_user_recommendations("u1", page=1, size=10)
    assert out["total"] == 2
    ids = [j["id"] for j in out["jobs"]]
    assert ids == ["j1", "j2"]

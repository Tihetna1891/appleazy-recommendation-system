import json
import pytest

import app.api.routes as routes

def test_recommendations_endpoint_cache_hit(client, monkeypatch):
    dummy_user_id = "user-cache"

    from app.services import recommender

    # Monkeypatch full pipeline
    monkeypatch.setattr(recommender, "fetch_user", lambda uid: {"resume": "mock resume"})
    monkeypatch.setattr(recommender, "fetch_jobs_open", lambda: [{"id": "job1", "description": "great job"}])
    monkeypatch.setattr(recommender, "embed_user", lambda u: [0.1, 0.2, 0.3])
    monkeypatch.setattr(recommender, "embed_jobs", lambda jobs: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(recommender, "store_jobs", lambda *a, **k: None)
    monkeypatch.setattr(recommender, "store_recommendations", lambda *a, **k: None)
    monkeypatch.setattr(recommender, "get_recommendations", lambda uid: [{"id": "job1", "score": 0.95}])
    
    #  Correct ID return
    monkeypatch.setattr(recommender, "get_job_by_id", lambda job_id: {
        "id": "job1",  # This ensures the test will get 'job1'
        "title": "Mock Job",
        "description": "Test Desc"
    })

    res = client.get(f"/recommendations/{dummy_user_id}?page=1&size=20")

    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert data["jobs"][0]["id"] == "job1"

from app.services import embedder

def test_embed_user_fallback(monkeypatch):
    # --- make embed() deterministic
    monkeypatch.setattr(embedder, "embed", lambda txt: [42.0])

    user = {
        "id": "u",
        "resume": "",
        "position": "Dev",
        "preferredIndustry": "Tech",
        "ethnicity": "N/A",
        "location": "Earth",
    }
    vec = embedder.embed_user(user)
    assert vec == [42.0]

Job Matching Backend – Developer Guide
This repo contains the FastAPI backend for the job recommendation system. Below are quick instructions to get started with local development, testing, and deployment.

 Setup
1. Clone and install dependencies
```cd job-matching ```
```python3 -m venv .venv ```
```source .venv/bin/activate ```
```pip install -r requirements.txt ```
2. Environment setup
Create a .env file with: the env.examplle

Running the App
Option 1: Docker (preferred)
```docker compose up --build ```
Runs both Redis and the FastAPI app.

Option 2: Manual
Start Redis separately, then:
```uvicorn app.main:app --reload ```


 API Endpoints
GET /recommendations/{user_id} – fetch exising recommendations for a specific user or if there is no recommendation for the user generate recommendations


Running Tests
```PYTHONPATH=. pytest -v ```
Tests are in tests/. You can run a specific test file:
``` pytest tests/test_recommender.py ```

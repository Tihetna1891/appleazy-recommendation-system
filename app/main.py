# main.py
from fastapi import FastAPI
from app.api.routes import router as api_router
from app.services.scheduler import start_scheduler

app = FastAPI(title="Job Recommender")

# Register routes
app.include_router(api_router)

# Start scheduler
start_scheduler()

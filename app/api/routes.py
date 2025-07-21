# api/routes.py
from fastapi import APIRouter, HTTPException, Query
from app.services.recommender import get_user_recommendations, get_all_recommendations, generate_recommendation_for_user
import traceback

router = APIRouter()
@router.get("/ping")
def ping():
    return {"status": "ok working..."}
@router.get("/recommendations/{user_id}")
def recommendations(user_id: str, page: int = Query(1, gt=0), size: int = Query(20, gt=0)):
    try:
        return get_user_recommendations(user_id, page, size)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(500, "Internal server error")
# api/routes.py
from fastapi import APIRouter, HTTPException, Query
from services.recommender import get_user_recommendations, get_all_recommendations, generate_recommendation_for_user
import traceback

router = APIRouter()

@router.get("/recommendations/{user_id}")
def recommendations(user_id: str, page: int = Query(1, gt=0), size: int = Query(20, gt=0)):
    try:
        return get_user_recommendations(user_id, page, size)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(500, "Internal server error")

@router.get("/recommendations")
def recommendations_for_all():
    try:
        return get_all_recommendations()
    except Exception as e:
        print("Error in all recommendations:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.post("/trigger-recommendation/{user_id}")
def trigger_recommendation(user_id: str):
    try:
        generate_recommendation_for_user(user_id)
        # Fetch recommendations immediately after generating
        recommendations = get_user_recommendations(user_id, page=1, size=100)  # Adjust size if needed
        return {
            # "status": "ok",
            "user_id": user_id,
            "recommendations": recommendations["jobs"]
        }
    except Exception as e:
        print("error", e)
        raise HTTPException(status_code=500, detail=str(e))
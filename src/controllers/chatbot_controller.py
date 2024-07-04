from fastapi import APIRouter, HTTPException
from src.services.chatbot_service import process_query

router = APIRouter()

@router.get("/chatbot/")
def ask_question(question: str):
    try:
        response = process_query(question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

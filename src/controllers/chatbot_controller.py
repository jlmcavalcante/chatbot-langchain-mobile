from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.chatbot_service import process_query

class QuestionRequest(BaseModel):
    question: str

router = APIRouter()

@router.post("/chatbot/")
async def ask_question(request: QuestionRequest):
    try:
        response = process_query(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

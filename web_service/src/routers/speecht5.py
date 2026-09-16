from fastapi import APIRouter, HTTPException
from src.services.speecht5_service import speak

router = APIRouter()


@router.post("/")
def create_wav(text: str):
    try:
        return speak(text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

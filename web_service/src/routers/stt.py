from fastapi import APIRouter, File, UploadFile, HTTPException
import tempfile, os, shutil
from src.services.stt_service import transcribe

router = APIRouter(prefix="/stt", tags=["STT"])

@router.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Le fichier doit être un audio")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        return transcribe(tmp_path)
    except Exception as e:
        raise HTTPException(500, f"Erreur: {str(e)}")
    finally:
        os.remove(tmp_path)
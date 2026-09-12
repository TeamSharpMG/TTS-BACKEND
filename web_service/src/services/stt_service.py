# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import tempfile
import os
import whisper
import shutil

# Variable globale pour le modèle
stt_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    global stt_model
    print("Chargement du modèle STT...")
    stt_model = whisper.load_model("base")  # ou ton modèle custom
    print("Modèle chargé ✅")
    yield
    # === Shutdown ===
    stt_model = None
    print("Modèle déchargé")

app = FastAPI(title="STT API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": stt_model is not None}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Vérification du type de fichier
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Le fichier doit être un audio")

    # Sauvegarde temporaire
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = stt_model.transcribe(tmp_path)
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
        }
    except Exception as e:
        raise HTTPException(500, f"Erreur de transcription: {str(e)}")
    finally:
        os.remove(tmp_path)
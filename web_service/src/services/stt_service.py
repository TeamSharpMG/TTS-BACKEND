# src/services/stt_service.py
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "ai" / "stt" / "model" # 📌 adapte ce chemin
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print(f"Chargement STT depuis {MODEL_DIR}")
        _processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSpeechSeq2Seq.from_pretrained(str(MODEL_DIR)).to(DEVICE).eval()
    return _model, _processor


def transcribe(audio_path: str) -> dict:
    model, processor = _load()

    # Chargement audio (soundfile) + resample (librosa) → pas besoin de torchaudio
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(**inputs)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text.strip()}
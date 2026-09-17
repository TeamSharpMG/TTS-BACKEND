import os

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import SpeechT5ForSpeechToText, SpeechT5Processor

from configuration import MODEL_PATHS

MODEL_DIR = MODEL_PATHS["stt_speecht5_model_2"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 450

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        print(f"Chargement STT depuis {MODEL_DIR}")
        _processor = SpeechT5Processor.from_pretrained(str(MODEL_DIR))
        _model = SpeechT5ForSpeechToText.from_pretrained(str(MODEL_DIR)).to(DEVICE)
        _model.eval()
        _model.gradient_checkpointing_disable()
        _model.config.use_cache = True
    return _model, _processor


def transcribe(audio_path: str) -> dict:
    model, processor = _load()

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(
            np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000
        )

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            min_length=2,
            num_beams=4,
        )

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return {"text": text.strip()}
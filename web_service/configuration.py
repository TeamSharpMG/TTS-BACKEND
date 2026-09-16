from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

MODEL_PATHS = {
    "mms_model": BASE_DIR / "ai" / "mms" / "mms_model",
    "mms_tokenizer": BASE_DIR / "ai" / "mms" / "mms_tokenizer",
    "stt_speecht5_model": BASE_DIR / "ai" / "stt_speecht5" / "model"
}

print(MODEL_PATHS)

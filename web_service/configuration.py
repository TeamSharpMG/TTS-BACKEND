from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

MODEL_PATHS = {
    "mms_model": BASE_DIR / "ai" / "mms" / "mms_model",
    "mms_tokenizer": BASE_DIR / "ai" / "mms" / "mms_tokenizer",
}

print(MODEL_PATHS)

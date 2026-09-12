# model_loader.py
import whisper  # ou ton framework

class STTModel:
    _instance = None

    @classmethod
    def load(cls, model_path: str = "base"):
        if cls._instance is None:
            print("Chargement du modèle STT...")
            cls._instance = whisper.load_model(model_path)
        return cls._instance
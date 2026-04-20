import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wav
import base64
from configuration import MODEL_PATHS, BASE_DIR
#TODO: factoriser en differents modules pour avoir du cleancode 

def speak(text:str):
    # 1. importer les modeles
    model = VitsModel.from_pretrained(MODEL_PATHS["mms_model"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["mms_tokenizer"])

    # 2. utiliser les modeles sur l'input
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform

    # 3. generer le fichier wav
    audio_numpy = output.squeeze().numpy()
    frequence = model.config.sampling_rate

    output_file = BASE_DIR / "output" / "out.wav"
    wav.write(output_file, rate=frequence, data=audio_numpy)

    # 4. envayer le wav 
    with open(output_file, "rb") as audio_file:
        audio = audio_file.read()

    audio_b64 = base64.b64encode(audio)
    audio_b64 = audio_b64.decode('ascii')

    return {
        "Modele":"mms",
        "audio":"wav-to-base64",
        "output": audio_b64
    }
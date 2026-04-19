from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello":"world"}

@app.post("/model/mms")
def read_mms(text: str):
    import torch
    from transformers import VitsModel, AutoTokenizer
    import scipy.io.wavfile as wav
    from configuration import MODEL_PATHS, BASE_DIR
    #TODO: factoriser en differents modules pour avoir du cleancode 

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

    output_file = "out.wav"
    wav.write(BASE_DIR / "output" / "out.wav", rate=frequence, data=audio_numpy)
    # 4. envayer le wav 

    return {
        "Modele":"mms",
        "audio":"base64",
        "input": text
    }
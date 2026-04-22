from fastapi import FastAPI
from src.routers import mms #,create_wav
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost:5173",  # your frontend (React, Vue, etc.)
    "http://127.0.0.1:5173",
    # your website
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello":"world"}

# @app.post("/model/mms")
# def read_mms(text: str):
    # import torch
    # from transformers import VitsModel, AutoTokenizer
    # import scipy.io.wavfile as wav
    # import base64
    # from configuration import MODEL_PATHS, BASE_DIR
    # #TODO: factoriser en differents modules pour avoir du cleancode 

    # # 1. importer les modeles
    # model = VitsModel.from_pretrained(MODEL_PATHS["mms_model"])
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["mms_tokenizer"])
    
    # # 2. utiliser les modeles sur l'input
    # inputs = tokenizer(text, return_tensors="pt")
    # with torch.no_grad():
    #     output = model(**inputs).waveform

    # # 3. generer le fichier wav
    # audio_numpy = output.squeeze().numpy()
    # frequence = model.config.sampling_rate

    # output_file = BASE_DIR / "output" / "out.wav"
    # wav.write(output_file, rate=frequence, data=audio_numpy)

    # # 4. envayer le wav 
    # with open(output_file, "rb") as audio_file:
    #     audio = audio_file.read()

    # audio_b64 = base64.b64encode(audio)
    # audio_b64 = audio_b64.decode('ascii')

    # return {
    #     "Modele":"mms",
    #     "audio":"wav-to-base64",
    #     "output": audio_b64
    # }
    # return create_wav(text)
    # ...

app.include_router(mms.router, prefix='/model/mms', tags=["model/mms"])
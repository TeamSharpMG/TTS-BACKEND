import base64
import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import torch
from configuration import BASE_DIR, MODEL_PATHS
from datasets import Dataset
from speechbrain.inference.speaker import EncoderClassifier
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

# TODO: factoriser en differents modules pour avoir du cleancode


def speak(text: str):
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
    audio_b64 = audio_b64.decode("ascii")

    return {"Modele": "mms", "audio": "wav-to-base64", "output": audio_b64}

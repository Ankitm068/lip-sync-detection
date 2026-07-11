from pathlib import Path

import librosa
import torch

# pyrefly: ignore [missing-import]
from src.speech.wav2vec_model import Wav2VecModel


class SpeechFeatureExtractor:

    def __init__(self, audio_path):

        self.audio_path = Path(audio_path)

        self.wav2vec = Wav2VecModel()

    def extract(self):

        audio, sr = librosa.load(
            self.audio_path,
            sr=16000,
            mono=True,
        )

        inputs = self.wav2vec.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():

            outputs = self.wav2vec.model(
                inputs.input_values
            )

        embeddings = outputs.last_hidden_state

        print(
            f"Embedding Shape : {embeddings.shape}"
        )

        return embeddings


if __name__ == "__main__":

    extractor = SpeechFeatureExtractor(
        "data/audio/vid1.wav"
    )

    extractor.extract()
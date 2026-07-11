from pathlib import Path
import librosa


class AudioLoader:
    """
    Load a WAV audio file into memory.
    """

    def __init__(self, audio_path: str):
        self.audio_path = Path(audio_path)

        if not self.audio_path.exists():
            raise FileNotFoundError(
                f"Audio not found: {self.audio_path}"
            )

    def load(self):

        audio, sample_rate = librosa.load(
            self.audio_path,
            sr=16000,
            mono=True
        )

        print(f"Sample Rate : {sample_rate}")
        print(f"Samples     : {len(audio)}")

        return audio, sample_rate


if __name__ == "__main__":

    loader = AudioLoader(
        "data/audio/vid1.wav"
    )

    audio, sr = loader.load()
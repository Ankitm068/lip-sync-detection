from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.audio.audio_energy import AudioEnergyExtractor
from src.audio.mel_spectrogram import MelBandEnergyExtractor
from src.speech.speech_features import SpeechFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeechSignalGenerator:
    """
    Builds a 1D speech signal to compare against mouth opening.

    method="mel_band" (recommended default):
        Sums Mel spectrogram power in the speech articulation band
        (≈ 300–3400 Hz). This tracks *phoneme* activity rather than raw
        volume — the signal rises when the jaw opens for vowels and
        drops during lip closures, making it far more correlated with
        mouth movement than RMS energy.

    method="energy":
        RMS energy envelope from the raw waveform. Simple and fast but
        measures loudness, not articulation. Fails on whispered speech
        (loud RMS but full mouth movement) and sustained hums (high RMS
        but closed lips).

    """

   
    _NATIVE_RATE_HZ = 50.0

    def __init__(
        self,
        audio_path: str,
        output_csv: str,
        method: str = "energy",
        frame_rate_hz: float = 50.0,
    ):

        self.audio_path = Path(audio_path)
        self.output_csv = Path(output_csv)
        self.method = method
        self.frame_rate_hz = frame_rate_hz

        if method not in ("energy", "wav2vec_norm", "mel_band"):
            raise ValueError(
                f"Unknown method '{method}'; expected "
                "'energy', 'mel_band', or 'wav2vec_norm'."
            )

        self.output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _resample_to_rate(
        signal: np.ndarray,
        native_rate_hz: float,
        target_rate_hz: float,
    ) -> np.ndarray:
        """
        Resample a 1D signal from its native frame rate to an exact
        target frame rate via linear interpolation.

        This guarantees the returned signal's length corresponds to
        `target_rate_hz` precisely, regardless of any hop-length
        rounding that happened upstream when the signal was extracted.
        """
        if abs(native_rate_hz - target_rate_hz) < 1e-4:
            return signal

        duration = (len(signal) - 1) / native_rate_hz
        x_old = np.linspace(0, duration, len(signal))
        n_new = int(round(duration * target_rate_hz)) + 1
        x_new = np.linspace(0, duration, n_new)

        interpolator = interp1d(
            x_old,
            signal,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        return interpolator(x_new)

    def _generate_energy(self):

        extractor = AudioEnergyExtractor(
            self.audio_path,
            frame_rate_hz=self.frame_rate_hz,
        )

        signal = extractor.extract()

        # Already extracted at self.frame_rate_hz
        return signal

    def _generate_mel_band(self):
        """Mel-band energy in the speech articulation band (≈ 300–3400 Hz).

        Returns a signal where each value is the summed log-power across
        the mel bins spanning that frequency band, in dB relative to the
        loudest frame. This tracks phoneme activity (jaw open/close)
        rather than volume.
        """
        extractor = MelBandEnergyExtractor(
            self.audio_path,
            frame_rate_hz=self.frame_rate_hz,
        )

        signal = extractor.extract()

        # Already extracted at self.frame_rate_hz
        return signal

    def _generate_wav2vec_norm(self):

        extractor = SpeechFeatureExtractor(
            self.audio_path
        )

        embeddings = extractor.extract()

        embeddings = embeddings.squeeze(0)

        embeddings = embeddings.cpu().numpy()

        norm = np.linalg.norm(
            embeddings,
            axis=1,
        )

        # wav2vec2 natively outputs at exactly 50 Hz.
        return self._resample_to_rate(
            norm, self._NATIVE_RATE_HZ, self.frame_rate_hz
        )

    def generate(self):

        if self.method == "mel_band":
            speech_signal = self._generate_mel_band()
        elif self.method == "energy":
            speech_signal = self._generate_energy()
        else:
            speech_signal = self._generate_wav2vec_norm()

        df = pd.DataFrame(
            {
                "time_step": np.arange(
                    len(speech_signal)
                ),
                "speech_signal": speech_signal,
            }
        )

        df.to_csv(
            self.output_csv,
            index=False,
        )

        logger.info(f"Speech Signal Generated ({self.method})")
        logger.info(f"Time Steps : {len(speech_signal)}")
        logger.info(f"CSV Saved  : {self.output_csv}")

        return speech_signal


if __name__ == "__main__":

    generator = SpeechSignalGenerator(
        audio_path="data/audio/vid1.wav",
        output_csv="data/output/speech_signal.csv",
        method="energy",
    )

    generator.generate()
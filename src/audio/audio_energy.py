from pathlib import Path

import librosa

from src.audio.audio_loader import AudioLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioEnergyExtractor:
    """
    Computes an RMS energy envelope directly from the audio waveform.

    This is a more direct proxy for "how much sound is happening right
    now" than wav2vec embedding magnitude is -- mouth opening should
    track energy/loudness far more reliably than it tracks the L2 norm
    of a transformer's internal hidden state, which reflects general
    activation magnitude rather than acoustic energy.

    Framed at the same rate as the wav2vec2-base output (50 Hz, i.e.
    one value every 20ms) so it lines up with the rest of the pipeline
    without needing a separate resample step.
    """

    def __init__(
        self,
        audio_path: str,
        frame_rate_hz: float = 50.0,
    ):
        self.audio_path = Path(audio_path)
        self.frame_rate_hz = frame_rate_hz

    def extract(self):

        loader = AudioLoader(self.audio_path)
        audio, sample_rate = loader.load()

        hop_length = int(round(sample_rate / self.frame_rate_hz))

        # Window a bit wider than the hop so consecutive frames overlap
        # slightly, which smooths the envelope without blurring it much.
        frame_length = hop_length * 2

        logger.debug(
            "Computing RMS energy: hop_length=%d frame_length=%d @ %.1f Hz",
            hop_length, frame_length, self.frame_rate_hz,
        )

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]

        logger.debug("RMS frames: %d  min/max: %.4f/%.4f", len(rms), rms.min(), rms.max())

        return rms


if __name__ == "__main__":

    extractor = AudioEnergyExtractor(
        audio_path="data/audio/vid1.wav",
    )

    energy = extractor.extract()

    logger.info("Frames : %d", len(energy))
    logger.info("Min/Max : %.4f / %.4f", energy.min(), energy.max())

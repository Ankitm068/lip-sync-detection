from pathlib import Path

import librosa
import numpy as np

from src.audio.audio_loader import AudioLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mel spectrogram configuration
# ---------------------------------------------------------------------------

# Output frame rate — must match the rest of the pipeline (wav2vec2 stride).
_FRAME_RATE_HZ = 50.0           # one speech frame every 20 ms

# Mel filter bank
_N_MELS = 128                   # total mel bins
_N_FFT  = 1024                  # FFT window (64 ms at 16 kHz)


# librosa.mel_frequencies rather than assumed.)
_FREQ_LO_HZ = 300.0    # low end of telephone-band speech
_FREQ_HI_HZ = 3400.0   # upper formant region (covers F1-F3 for vowels)


# compare −∞ values downstream.
_DB_FLOOR = -80.0


def _hz_to_mel_bin(freq_hz: float, n_mels: int, sample_rate: float, fmax: float) -> int:
    """
    Convert a target frequency (Hz) to the nearest mel filter-bank bin
    index, for the given n_mels / sample_rate / fmax configuration.

    Uses librosa's own mel_frequencies so the mapping always matches
    whatever librosa.feature.melspectrogram actually produces — no
    hand-derived approximations that can drift out of sync.
    """
    # n_mels + 2 gives the filter *edges*; index i+1 is the center of bin i.
    mel_edges = librosa.mel_frequencies(
        n_mels=n_mels + 2,
        fmin=0.0,
        fmax=fmax,
    )
    bin_centers = mel_edges[1:-1]  # length n_mels, one center per bin

    return int(np.argmin(np.abs(bin_centers - freq_hz)))


class MelBandEnergyExtractor:
    

    def __init__(
        self,
        audio_path: str,
        frame_rate_hz: float = _FRAME_RATE_HZ,
        freq_lo_hz: float = _FREQ_LO_HZ,
        freq_hi_hz: float = _FREQ_HI_HZ,
        mel_bin_lo: int | None = None,
        mel_bin_hi: int | None = None,
    ):
        self.audio_path    = Path(audio_path)
        self.frame_rate_hz = frame_rate_hz
        self.freq_lo_hz    = freq_lo_hz
        self.freq_hi_hz    = freq_hi_hz
        self.mel_bin_lo    = mel_bin_lo
        self.mel_bin_hi    = mel_bin_hi

    def extract(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray, shape (n_frames,)
            Per-frame log power in the speech band, in dB, at
            `frame_rate_hz`. Values are clamped to [_DB_FLOOR, 0] dB.
        """
        logger.info("Extracting mel-band energy from: %s", self.audio_path.name)
        loader = AudioLoader(self.audio_path)
        audio, sample_rate = loader.load()
        logger.info("Audio loaded — %.1f s @ %d Hz", len(audio) / sample_rate, sample_rate)

      
        hop_length   = int(round(sample_rate / self.frame_rate_hz))
        frame_length = _N_FFT                 # ~64 ms analysis window
        fmax         = sample_rate / 2

        logger.debug(
            "Mel spectrogram: hop=%d n_fft=%d fmax=%.0f Hz",
            hop_length, frame_length, fmax,
        )

        # ── Mel spectrogram ───────────────────────────────────────────
        # Shape: (n_mels, n_frames)  — power (amplitude²) in each bin.
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=_N_MELS,
            fmin=0.0,
            fmax=fmax,
            center=True,
        )

        # ── Resolve articulation band to bin indices ──────────────────
       
        if self.mel_bin_lo is not None and self.mel_bin_hi is not None:
            lo, hi = self.mel_bin_lo, self.mel_bin_hi
        else:
            lo = _hz_to_mel_bin(self.freq_lo_hz, _N_MELS, sample_rate, fmax)
            hi = _hz_to_mel_bin(self.freq_hi_hz, _N_MELS, sample_rate, fmax)

        # Clamp indices to valid range in case caller overrides defaults.
        lo = max(0, min(lo, _N_MELS - 1))
        hi = max(lo + 1, min(hi, _N_MELS))

        logger.debug(
            "Speech band: %.0f–%.0f Hz → mel bins [%d:%d]",
            self.freq_lo_hz, self.freq_hi_hz, lo, hi,
        )

        band = mel_spec[lo:hi, :]          # shape: (n_band_bins, n_frames)

        # ── Sum power across selected bins ────────────────────────────
        band_power = band.sum(axis=0)      # shape: (n_frames,)

        # ── Convert to dB ─────────────────────────────────────────────
        
        band_db = librosa.power_to_db(
            band_power,
            ref=np.max(band_power) if band_power.max() > 0 else 1.0,
            top_db=abs(_DB_FLOOR),
        )

        logger.debug(
            "Mel-band frames: %d  min/max: %.2f / %.2f dB",
            len(band_db), band_db.min(), band_db.max(),
        )

        return band_db


if __name__ == "__main__":

    extractor = MelBandEnergyExtractor(
        audio_path="data/audio/vid1.wav",
    )

    signal = extractor.extract()

    logger.info("Frames   : %d", len(signal))
    logger.info("Min / Max: %.2f dB / %.2f dB", signal.min(), signal.max())
"""
src.audio — Audio Extraction & Speech-Signal Sub-package
=========================================================

Exposes all audio-related classes so callers can import directly from
the sub-package instead of knowing the internal module layout::

    from src.audio import AudioExtractor, AudioLoader
    from src.audio import AudioEnergyExtractor, MelBandEnergyExtractor

Classes
-------
AudioExtractor
    Extracts the audio track from a video and writes a 16 kHz mono WAV file
    using the bundled FFmpeg binary (no system-wide FFmpeg install needed).

AudioLoader
    Loads a WAV file into a NumPy array at 16 kHz via librosa.

AudioEnergyExtractor
    Computes an RMS energy envelope (loudness proxy) at a configurable frame
    rate.  Available as ``method="energy"`` in the pipeline but not the
    default — it measures loudness, not articulation.

MelBandEnergyExtractor
    Sums log-power across the 300–3400 Hz mel-filter bands at a configurable
    frame rate.  This is the default speech signal because it tracks phoneme/
    articulation activity rather than raw volume.
"""

from src.audio.audio_extractor import AudioExtractor
from src.audio.audio_loader import AudioLoader
from src.audio.audio_energy import AudioEnergyExtractor
from src.audio.mel_spectrogram import MelBandEnergyExtractor

__all__ = [
    "AudioExtractor",
    "AudioLoader",
    "AudioEnergyExtractor",
    "MelBandEnergyExtractor",
]

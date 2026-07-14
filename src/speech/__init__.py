"""
src.speech — Speech Feature & Signal Sub-package
=================================================

Exposes all speech-related classes so callers can import directly from
the sub-package::

    from src.speech import SpeechSignalGenerator
    from src.speech import SpeechFeatureExtractor, Wav2VecModel

Classes
-------
SpeechSignalGenerator
    Main pipeline component.  Routes to one of three backends based on the
    ``method`` parameter and writes the resulting 1-D signal to a CSV:

    - ``"mel_band"``      — Mel-band energy (default, recommended)
    - ``"energy"``        — RMS energy envelope (legacy)
    - ``"wav2vec_norm"``  — L2-norm of wav2vec2 embeddings (abandoned)

SpeechFeatureExtractor
    Loads and runs ``facebook/wav2vec2-base-960h`` via Hugging Face
    Transformers to extract hidden-state embeddings.  Used internally by
    ``SpeechSignalGenerator`` when ``method="wav2vec_norm"``.

Wav2VecModel
    Thin wrapper that loads ``Wav2Vec2Processor`` and ``Wav2Vec2Model``
    from Hugging Face and calls ``model.eval()``.
"""

from src.speech.speech_signal import SpeechSignalGenerator
from src.speech.speech_features import SpeechFeatureExtractor
from src.speech.wav2vec_model import Wav2VecModel

__all__ = [
    "SpeechSignalGenerator",
    "SpeechFeatureExtractor",
    "Wav2VecModel",
]

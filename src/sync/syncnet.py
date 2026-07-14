"""
src/sync/syncnet.py — SyncNet-inspired audio-visual synchronisation stub.

References the SyncNet paper (Chung & Zisserman, "Out of Time: Automated
Lip Sync in the Wild", ACCV 2016).  The current pipeline uses a classical
signal-correlation approach (NCC + Pearson) instead of a trained network,
but this class provides the interface a future deep-learning implementation
would expose so it can be swapped in transparently.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyncNetAnalyzer:
    """
    Placeholder for a SyncNet-based audio-visual synchronisation analyser.

    In the original SyncNet architecture, a twin-stream convolutional network
    is trained to embed mouth-region video clips and MFCC audio windows into a
    shared metric space.  The distance between the two embeddings at each
    time-step measures sync quality without needing explicit signal alignment.

    Current status
    --------------
    Not yet implemented.  The pipeline currently uses the classical
    :class:`~src.sync.signal_alignment.SignalAlignment` +
    :class:`~src.sync.correlation.CorrelationAnalyzer` approach, which does
    not require a trained model and works well for the interview-fraud
    detection use-case.

    This class exists so that:
    1. The ``src.sync`` package has a named export for SyncNet.
    2. A future implementation can drop in here and be picked up by the
       ``__init__.py`` without any other changes.

    Parameters
    ----------
    model_path:
        Path to the pre-trained SyncNet weights file (not required while
        the class is a stub).
    threshold:
        Confidence threshold below which a video segment is flagged as
        out-of-sync (range 0–1).
    """

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.threshold  = threshold
        logger.debug(
            "SyncNetAnalyzer initialised (stub) — model_path=%s threshold=%.2f",
            model_path, threshold,
        )

    def load_model(self) -> None:
        """
        Load the pre-trained SyncNet model weights.

        Raises
        ------
        NotImplementedError
            Always — this method is not yet implemented.
        """
        raise NotImplementedError(
            "SyncNetAnalyzer.load_model() is not yet implemented. "
            "Use SignalAlignment + CorrelationAnalyzer for sync scoring."
        )

    def analyse(self, video_path: str, audio_path: str) -> dict:
        """
        Run the SyncNet sync-offset analysis on a video/audio pair.

        Parameters
        ----------
        video_path:
            Path to the input video file (mouth-region crop expected).
        audio_path:
            Path to the corresponding audio file (16 kHz mono WAV).

        Returns
        -------
        dict
            Would contain ``{"offset_ms": float, "confidence": float}``.

        Raises
        ------
        NotImplementedError
            Always — this method is not yet implemented.
        """
        raise NotImplementedError(
            "SyncNetAnalyzer.analyse() is not yet implemented. "
            "Use SignalAlignment + CorrelationAnalyzer for sync scoring."
        )

"""
src.sync — Signal Alignment & Synchronisation Scoring Sub-package
=================================================================

Exposes all synchronisation-related classes so callers can import
directly from the sub-package::

    from src.sync import SignalAlignment, CorrelationAnalyzer, SyncDecision
    from src.sync import SyncNetAnalyzer  # stub — not yet implemented

Classes
-------
SignalAlignment
    Phase-3 pipeline step.  Takes the raw mouth-feature CSV and speech-
    signal CSV, runs the full 8-step alignment pipeline (interpolate NaN
    frames → resample → robust-normalise → Savitzky-Golay smooth →
    mouth-delta blend → FFT-accelerated NCC lag search → apply lag →
    save aligned CSVs + JSON meta).

CorrelationAnalyzer
    Phase-4 pipeline step.  Reads the aligned CSVs and computes Pearson
    correlation, cosine similarity, and RMSE.  Only Pearson is used for
    the final lip-sync score; the others are saved for diagnostics.

SyncDecision
    Phase-5 pipeline step.  Converts the lip-sync score (0–100) into a
    human-readable verdict (SYNCED / LIKELY_SYNCED / UNCERTAIN / NOT_SYNCED)
    and applies reliability penalties (suspicious lag, too many missing frames).

SyncNetAnalyzer
    Stub for a future SyncNet-based deep-learning analyser.  Currently raises
    ``NotImplementedError`` on any call.
"""

from src.sync.signal_alignment import SignalAlignment
from src.sync.correlation import CorrelationAnalyzer
from src.sync.decision import SyncDecision
from src.sync.syncnet import SyncNetAnalyzer

__all__ = [
    "SignalAlignment",
    "CorrelationAnalyzer",
    "SyncDecision",
    "SyncNetAnalyzer",
]

"""
src — Lip-Sync Detection Package
=================================

Top-level package for the lip-sync detection pipeline.

Sub-packages
------------
- audio   : audio extraction and speech-signal generation
- face    : face detection, landmark tracking, and mouth-feature extraction
- speech  : speech feature extraction (wav2vec2, mel-band)
- sync    : signal alignment, correlation scoring, and sync decision
- utils   : shared config, logging, and file utilities
- video   : video frame extraction and loading

Quick import
------------
    from src.pipeline import run_pipeline
"""

from src.pipeline import run_pipeline

__all__ = ["run_pipeline"]

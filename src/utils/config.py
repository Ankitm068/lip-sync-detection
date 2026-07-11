"""
src/utils/config.py — Shared pipeline configuration constants.

Centralising these avoids magic numbers drifting out of sync between
pipeline.py, signal_alignment.py, and decision.py.
"""

# ── Speech signal timing ────────────────────────────────────────────────────
# The speech signal is generated at 50 Hz (one frame every 20 ms).
# This was reduced from 100 Hz to narrow the frequency gap between the
# speech envelope and the mouth-landmark signal (~30 FPS). At 100 Hz the
# speech signal captured fast phoneme-level bursts (plosive transients,
# ~5-10 ms) that the face landmarker physically cannot track at 33 ms/frame,
# producing a systematic temporal mismatch that drove Pearson correlation
# toward zero even on perfectly synced video.
SPEECH_SIGNAL_RATE_HZ = 30.0

# ── Speech envelope smoothing ────────────────────────────────────────────────
# Additional Savitzky-Golay smoothing window (ms) applied to the speech
# envelope *after* normalization in SignalAlignment. This further removes
# residual fast-phoneme oscillations the mouth signal cannot match,
# so that both signals operate at the same effective temporal bandwidth
# before cross-correlation and Pearson scoring.
SPEECH_ENVELOPE_SMOOTH_MS = 250.0

# ── Mouth delta blend weight ──────────────────────────────────────────────────
# Weight given to the first derivative (Δmouth) when blending with the
# absolute mouth signal.  The blended signal is:
#   mouth_blended = (1 - w) * mouth + w * mouth_delta_normalised
# The derivative captures jaw-opening *onsets* that align better with
# speech burst timing than the slow absolute position does.
MOUTH_DELTA_WEIGHT = 0.0

# ── Lag search ───────────────────────────────────────────────────────────────
# How far to search, in either direction, for the audio/video offset that
# best aligns the two signals.
MAX_LAG_SECONDS = 1.0

"""
src/pipeline.py — Shared async pipeline logic.

Executes the lip-sync detection steps in a clean pipeline.
"""

import asyncio
import json
import time
from pathlib import Path

from src.audio.audio_extractor import AudioExtractor
from src.face.mouth_features import MouthFeatureExtractor
from src.speech.speech_signal import SpeechSignalGenerator
from src.sync.correlation import CorrelationAnalyzer
from src.sync.decision import SyncDecision
from src.sync.signal_alignment import SignalAlignment
from src.utils import config
from src.utils.logger import get_logger
from src.video.frame_extractor import FrameExtractor

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Individual step coroutines
# ═══════════════════════════════════════════════════════════════════════════════


async def step_extract_frames(video_path: str, frames_dir: str) -> tuple[float, int]:
    """Phase 1a — Extract every video frame as .jpg, record FPS."""
    extractor = FrameExtractor(
        video_path=video_path,
        output_dir=frames_dir,
    )
    fps, frame_count = await asyncio.to_thread(extractor.extract_frames)
    return fps, frame_count


async def step_extract_audio(video_path: str, audio_path: str) -> None:
    """Phase 1b — Extract audio track as 16 kHz mono WAV."""
    extractor = AudioExtractor(video_path=video_path, output_audio=audio_path)
    await asyncio.to_thread(extractor.extract)


async def step_mouth_features(frames_dir: str, mouth_csv: str, fps: float) -> None:
    """Phase 2a — Detect face landmarks and compute normalized mouth opening.

    `fps` is passed through so the extractor can run MediaPipe in VIDEO mode
    (temporal tracking across frames, instead of re-detecting from scratch
    on every frame) which needs a per-frame timestamp to work.
    """
    extractor = MouthFeatureExtractor(input_dir=frames_dir, output_csv=mouth_csv, fps=fps)
    await asyncio.to_thread(extractor.process)


async def step_speech_signal(
    audio_path: str,
    speech_csv: str,
    method: str = "mel_band",
    speech_rate_hz: float = config.SPEECH_SIGNAL_RATE_HZ,
) -> None:
    """Phase 2b — Generate 1-D speech activity signal.

    Generated at `speech_rate_hz` (default 100 Hz) rather than the video's
    FPS — audio carries temporal detail (fast consonant transitions,
    ~50-150ms) that gets blurred if it's downsampled to ~30Hz too early.
    The mouth signal (smooth/continuous landmark data) is upsampled to meet
    it instead, in SignalAlignment.
    """
    generator = SpeechSignalGenerator(
        audio_path=audio_path,
        output_csv=speech_csv,
        method=method,
        frame_rate_hz=speech_rate_hz,
    )
    await asyncio.to_thread(generator.generate)


async def step_align(
    mouth_csv: str,
    speech_csv: str,
    aligned_dir: str,
    video_fps: float,
    speech_rate_hz: float = config.SPEECH_SIGNAL_RATE_HZ,
    max_lag_seconds: float = config.MAX_LAG_SECONDS,
    speech_envelope_smooth_ms: float = config.SPEECH_ENVELOPE_SMOOTH_MS,
    mouth_delta_weight: float = config.MOUTH_DELTA_WEIGHT,
) -> dict:
    """Phase 3 — Search lag, trim to overlap."""
    alignment = SignalAlignment(
        mouth_csv=mouth_csv,
        speech_csv=speech_csv,
        output_dir=aligned_dir,
        video_fps=video_fps,
        speech_rate_hz=speech_rate_hz,
        max_lag_seconds=max_lag_seconds,
        speech_envelope_smooth_ms=speech_envelope_smooth_ms,
        mouth_delta_weight=mouth_delta_weight,
    )
    await asyncio.to_thread(alignment.process)

    meta_path = Path(aligned_dir) / "alignment_result.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta


async def step_correlate(aligned_dir: str) -> dict:
    """Phase 4 — Compute correlation metrics between aligned signals."""
    analyzer = CorrelationAnalyzer(
        mouth_csv=str(Path(aligned_dir) / "aligned_mouth_signal.csv"),
        speech_csv=str(Path(aligned_dir) / "aligned_speech_signal.csv"),
        alignment_meta_json=str(Path(aligned_dir) / "alignment_result.json"),
    )
    correlation_report = await asyncio.to_thread(analyzer.compute)
    return correlation_report


async def step_decide(correlation_report: dict, meta: dict, output_path: str) -> dict:
    """Phase 5 — Map correlation to decision and save JSON."""
    decision = SyncDecision(
        lip_sync_score=correlation_report["lip_sync_score"],
        alignment_meta=meta,
        correlation_metrics=correlation_report.get("metrics"),
        max_lag_seconds=config.MAX_LAG_SECONDS,
    )
    result = await asyncio.to_thread(decision.report)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Top-level orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


async def run_pipeline(video_path: str, speech_method: str = "mel_band", job_id: str | None = None) -> dict:
    """
    Run the complete lip-sync detection pipeline.
    Returns a flat result dict suitable for JSON serialisation.
    """
    # ── Isolated paths ─────────────────────────────────────────────────────────
    stem          = job_id or Path(video_path).stem
    audio_path    = f"data/audio/{stem}.wav"
    frames_dir    = f"data/frames/{stem}"
    output_dir    = f"data/output/{stem}"
    aligned_dir   = f"{output_dir}/aligned"
    mouth_csv     = f"{output_dir}/mouth_signal.csv"
    speech_csv    = f"{output_dir}/speech_signal.csv"
    decision_path = f"{aligned_dir}/decision.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Lip-Sync pipeline for video: {video_path}")
    logger.info(f"Job ID: {stem}")

    start = time.perf_counter()

    # ── Phase 1 — parallel ─────────────────────────────────────────────────────
    logger.info("Phase 1: Extracting frames and audio...")
    results = await asyncio.gather(
        step_extract_frames(video_path, frames_dir),
        step_extract_audio(video_path, audio_path),
    )
    fps, frame_count = results[0]

    # ── Phase 2 — parallel ─────────────────────────────────────────────────────
    logger.info("Phase 2: Extracting mouth features and speech signal...")
    await asyncio.gather(
        step_mouth_features(frames_dir, mouth_csv, fps),
        step_speech_signal(audio_path, speech_csv, speech_method, config.SPEECH_SIGNAL_RATE_HZ),
    )

    # ── Phases 3–5 — sequential ────────────────────────────────────────────────
    logger.info("Phase 3: Aligning signals...")
    meta        = await step_align(mouth_csv, speech_csv, aligned_dir, fps, config.SPEECH_SIGNAL_RATE_HZ)
    logger.info("Phase 4: Computing correlation...")
    correlation_report = await step_correlate(aligned_dir)
    logger.info("Phase 5: Making final decision...")
    result      = await step_decide(correlation_report, meta, decision_path)

    total_time = time.perf_counter() - start
    logger.info(f"Pipeline completed in {total_time:.1f}s with verdict: {result['verdict']}")

    return {
        "job_id":              stem,
        "verdict":             result["verdict"],
        "correlation":         round(float(correlation_report["lip_sync_score"]), 4),
        "best_lag_ms":         round(float(meta["best_lag_ms"]), 1),
        "reason":              result["reason"],
        "warnings":            result.get("warnings", []),
        "frame_count":         frame_count,
        "fps":                 round(float(fps), 2),
        "speech_method":       speech_method,
        "total_time_seconds":  round(total_time, 1),
        "decision_json_path":  decision_path,
    }
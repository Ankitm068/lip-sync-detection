import os

# ── Numba JIT bypass ───────────────────────────────────────────────────────────
# Windows Application Control policy blocks numba's compiled _dispatcher.pyd DLL.
# Setting NUMBA_DISABLE_JIT=1 makes numba skip JIT compilation and run in
# pure-Python fallback mode. librosa works correctly either way.
# Must be set BEFORE any librosa/numba import.
if not os.environ.get("NUMBA_DISABLE_JIT"):
    os.environ["NUMBA_DISABLE_JIT"] = "1"

"""
main.py — Simplified Lip-Sync Detection API
==========================================

Start the server:
    uv run uvicorn main:app --reload --port 8000

Endpoints:
    POST /upload             — Upload a video file
    POST /process/{video_id} — Run detection on the video synchronously
"""

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.pipeline import run_pipeline

logger = get_logger(__name__)

# ── Allowed video extensions ───────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Upload directory ───────────────────────────────────────────────────────────
UPLOAD_DIR = Path("data/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FastAPI app & Models
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Simple Lip-Sync Detection API",
    description="Minimal API to upload a video and run lip-sync detection synchronously.",
    version="1.0.0",
)


class UploadResponse(BaseModel):
    video_id: str = Field(..., description="Unique ID for the uploaded video.")
    message: str = Field(..., description="Success message.")


class ProcessResponse(BaseModel):
    job_id: str
    verdict: str
    correlation: float
    best_lag_ms: float
    reason: str
    warnings: list[str]
    frame_count: int
    fps: float
    speech_method: str
    total_time_seconds: float
    decision_json_path: str


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"status": "ok", "message": "Simple Lip-Sync API is running."}


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a video file for later processing",
)
async def upload_video(
    file: UploadFile = File(..., description="Video file (.mp4, .avi, etc).")
) -> UploadResponse:
    """
    Saves the video and returns a `video_id`.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"

    contents = await file.read()
    video_path.write_bytes(contents)

    logger.info("Video uploaded: id=%s filename=%s size=%d bytes", video_id, file.filename, len(contents))

    return UploadResponse(
        video_id=video_id,
        message=f"Video uploaded successfully. You can now call POST /process/{video_id}"
    )


@app.post(
    "/process/{video_id}",
    response_model=ProcessResponse,
    summary="Process the uploaded video synchronously",
)
async def process_video(video_id: str) -> ProcessResponse:
    """
    Runs the lip-sync detection pipeline.
    WARNING: This request will block for ~30 seconds while the ML models run.
    """
    # Find the video file by matching the video_id prefix in the upload directory
    matching_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    video_path = matching_files[0]
    logger.info("Processing request: video_id=%s path=%s", video_id, video_path)

    try:
        # Run the pipeline (this takes 30+ seconds)
        result = await run_pipeline(
            video_path=str(video_path),
            # "mel_band" tracks phoneme/articulation activity (jaw open/close)
            # rather than raw loudness, so it correlates with mouth movement
            # far more reliably than the RMS "energy" method.
            speech_method="mel_band",
            job_id=video_id
        )
        logger.info("Process complete: video_id=%s verdict=%s", video_id, result.get('verdict'))
        return ProcessResponse(**result)

    except Exception as exc:
        logger.error("Pipeline error for video_id=%s: %s", video_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

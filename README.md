# 🎙️ Lip-Sync Detection

> **Video Interview Fraud Detection using Computer Vision & Audio Analysis**

A research-grade system that detects lip-sync mismatch in videos by independently extracting and correlating mouth-movement signals (from face landmarks) with speech-activity signals (from audio spectrogram analysis). Designed for catching pre-recorded or deepfake video submissions in online interviews.

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Verdict System](#verdict-system)
- [Running Modules Individually](#running-modules-individually)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)

---

## Overview

This pipeline answers one question: **"Is the person in this video actually speaking, or is the audio dubbed/faked?"**

It does **not** use a trained deepfake classifier. Instead it takes a purely signal-processing approach:

1. Extract the speaker's **mouth-opening signal** from every video frame using MediaPipe's 478-point face landmark model
2. Extract a **speech-activity signal** from the audio using Mel spectrogram band energy (300–3400 Hz)
3. Time-align the two signals using **Normalized Cross-Correlation (NCC)**
4. Score them using **Pearson correlation** — a high score means the signals rise and fall together (in sync); a low score means they are unrelated (likely fake)

---

## How It Works

The pipeline runs in **5 phases**, with Phases 1 and 2 executing in parallel:

```
 Video File
     │
     ├── Phase 1a ──  FrameExtractor       → frame_000000.jpg … frame_N.jpg
     │                                        video_meta.json (fps, frame_count)
     │
     ├── Phase 1b ──  AudioExtractor       → 16 kHz mono WAV
     │
     ├── Phase 2a ──  MouthFeatureExtractor → mouth_signal.csv
     │               (MediaPipe FaceLandmarker · VIDEO mode)
     │               4 features per frame:
     │                 • mouth_height  (inner + outer lip gap / inter-ocular)
     │                 • inner_area    (shoelace area of 16-pt inner lip contour)
     │                 • velocity      (signed Δ mouth_height frame-to-frame)
     │                 • jaw_opening   (upper lip → chin / inter-ocular)
     │               → composite mouth_signal = weighted blend
     │
     ├── Phase 2b ──  SpeechSignalGenerator → speech_signal.csv
     │               (MelBandEnergyExtractor · 300–3400 Hz · dB scale)
     │
     ├── Phase 3  ──  SignalAlignment
     │               • Interpolate NaN frames (no-face gaps)
     │               • Resample both to shared 30 Hz timeline
     │               • Robust normalize (median-filter → percentile-clip → MinMax)
     │               • Savitzky-Golay smooth (mouth: 220 ms, speech: 250 ms)
     │               • Blend mouth with |Δmouth| (onset emphasis)
     │               • FFT-accelerated NCC lag search ± 400 ms
     │               • Apply best lag → trim to overlap
     │               → aligned_mouth_signal.csv, aligned_speech_signal.csv,
     │                 alignment_result.json
     │
     ├── Phase 4  ──  CorrelationAnalyzer
     │               • Pearson correlation  (used for scoring)
     │               • Cosine similarity    (diagnostics only)
     │               • RMSE                 (diagnostics only)
     │               → correlation_report.json, signal_overlay_plot.png
     │
     └── Phase 5  ──  SyncDecision
                     • Score → SYNCED / LIKELY_SYNCED / UNCERTAIN / NOT_SYNCED
                     • Reliability penalties (suspicious lag, too many missing frames)
                     → decision.json
```

---

## Project Structure

```
lip-sync-detection/
│
├── main.py                    # FastAPI server (entry point)
├── pyproject.toml             # Dependencies & project metadata
├── logging.yaml               # Dual-output logging configuration
├── .env                       # Environment variables (not committed)
│
├── src/
│   ├── __init__.py            # Exposes run_pipeline
│   ├── pipeline.py            # Async pipeline orchestrator
│   │
│   ├── audio/
│   │   ├── __init__.py        # Exposes all 4 audio classes
│   │   ├── audio_extractor.py # FFmpeg WAV extraction
│   │   ├── audio_loader.py    # librosa WAV loader (16 kHz mono)
│   │   ├── audio_energy.py    # RMS energy envelope (legacy)
│   │   └── mel_spectrogram.py # Mel-band energy (default)
│   │
│   ├── face/
│   │   ├── __init__.py        # Exposes FaceDetector, MouthFeatureExtractor, MouthMeshVisualizer
│   │   ├── face_detector.py   # BlazeFace bounding-box detector
│   │   ├── mouth_features.py  # Core: 4-feature mouth-signal extraction
│   │   └── face_mesh.py       # Debug visualizer: draws geometry on frames
│   │
│   ├── speech/
│   │   ├── __init__.py        # Exposes SpeechSignalGenerator, SpeechFeatureExtractor, Wav2VecModel
│   │   ├── speech_signal.py   # Router: mel_band / energy / wav2vec_norm
│   │   ├── speech_features.py # wav2vec2 embedding extractor (legacy)
│   │   └── wav2vec_model.py   # Hugging Face wav2vec2-base-960h wrapper
│   │
│   ├── sync/
│   │   ├── __init__.py        # Exposes SignalAlignment, CorrelationAnalyzer, SyncDecision, SyncNetAnalyzer
│   │   ├── signal_alignment.py  # 8-step NCC alignment pipeline
│   │   ├── correlation.py       # Pearson / cosine / RMSE scoring
│   │   ├── decision.py          # Score → verdict + reliability warnings
│   │   └── syncnet.py           # SyncNetAnalyzer stub (not yet implemented)
│   │
│   ├── utils/
│   │   ├── __init__.py        # Exposes LoggerFactory, get_logger, FileUtils, config
│   │   ├── config.py          # Shared pipeline constants
│   │   ├── logger.py          # LoggerFactory class + get_logger()
│   │   └── file_utils.py      # FileUtils class (ensure_dir, read_json, etc.)
│   │
│   └── video/
│       ├── __init__.py        # Exposes FrameExtractor, VideoLoader
│       ├── frame_extractor.py # OpenCV frame-by-frame extraction
│       └── video_loader.py    # Video info & preview utility
│
├── models/
│   ├── face_landmarker.task         # MediaPipe 478-point face mesh (3.6 MB)
│   └── blaze_face_short_range.tflite  # MediaPipe BlazeFace detector (224 KB)
│
├── data/
│   ├── videos/    # Uploaded video files (UUID-named)
│   ├── frames/    # Extracted frames (one subfolder per job)
│   ├── audio/     # Extracted WAV files
│   └── output/    # Per-job results: CSVs, JSONs, plots
│
└── logs/
    └── lip_sync.log   # Persistent debug log
```

---

## Installation

### Prerequisites

- **Python 3.12+**
- **`uv`** package manager

```powershell
# Install uv (if not already installed)
pip install uv
# or
winget install astral-sh.uv
```

### 1. Clone the repository

```powershell
git clone https://github.com/your-username/lip-sync-detection.git
cd lip-sync-detection
```

### 2. Install dependencies

```powershell
uv sync
```

### 3. Download MediaPipe models

```powershell
# Create models directory
New-Item -ItemType Directory -Force -Path models

# Download FaceLandmarker model (required for mouth feature extraction)
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile "models/face_landmarker.task"

# Download BlazeFace model (used by face_detector.py)
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" `
  -OutFile "models/blaze_face_short_range.tflite"
```

---

## Quick Start

### Start the API server

```powershell
uv run uvicorn main:app --reload --port 8000
```

### Upload a video and run detection

```powershell
# Step 1 — Upload your video
$resp = Invoke-RestMethod `
    -Uri "http://localhost:8000/upload" `
    -Method POST `
    -Form @{ file = Get-Item "path/to/your/video.mp4" }

$id = $resp.video_id
Write-Host "Video ID: $id"

# Step 2 — Run lip-sync detection (takes ~30 seconds)
$result = Invoke-RestMethod `
    -Uri "http://localhost:8000/process/$id" `
    -Method POST

$result | ConvertTo-Json
```

### Example response

```json
{
  "job_id": "3f7a2b1c-8e9d-4f12-a3b4-c5d6e7f8a9b0",
  "verdict": "SYNCED",
  "correlation": 0.7832,
  "best_lag_ms": -20.0,
  "reason": "Lip Sync Score indicates excellent synchronization.",
  "warnings": [],
  "frame_count": 900,
  "fps": 30.0,
  "speech_method": "mel_band",
  "total_time_seconds": 28.4,
  "decision_json_path": "data/output/3f7a2b1c-.../aligned/decision.json"
}
```

### Interactive API docs

Open **http://localhost:8000/docs** in your browser for Swagger UI.

---

## API Reference

### `GET /`

Health check.

**Response:**
```json
{ "status": "ok", "message": "Simple Lip-Sync API is running." }
```

---

### `POST /upload`

Upload a video file. Returns a `video_id` to use in the next call.

**Request:** `multipart/form-data` with field `file`

**Supported formats:** `.mp4` `.avi` `.mov` `.mkv` `.webm`

**Response:**
```json
{
  "video_id": "3f7a2b1c-...",
  "message": "Video uploaded successfully. You can now call POST /process/3f7a2b1c-..."
}
```

**Errors:**
| Code | Reason |
|------|--------|
| `415` | Unsupported file extension |

---

### `POST /process/{video_id}`

Run the full 5-phase lip-sync detection pipeline on an uploaded video.

> ⚠️ **This endpoint blocks for ~30 seconds** while the ML models process the video. Plan accordingly if integrating into a UI.

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | UUID for this job |
| `verdict` | string | `SYNCED` / `LIKELY_SYNCED` / `UNCERTAIN` / `NOT_SYNCED` |
| `correlation` | float | Pearson correlation score (0–1) |
| `best_lag_ms` | float | Audio/video offset found by NCC (ms) |
| `reason` | string | Human-readable explanation of the verdict |
| `warnings` | array | Reliability warnings (suspicious lag, missing frames) |
| `frame_count` | int | Total frames extracted from the video |
| `fps` | float | Video frame rate |
| `speech_method` | string | Signal method used (always `mel_band`) |
| `total_time_seconds` | float | Wall-clock pipeline duration |
| `decision_json_path` | string | Path to the saved `decision.json` |

**Errors:**
| Code | Reason |
|------|--------|
| `404` | `video_id` not found |
| `500` | Pipeline error (check `logs/lip_sync.log` for details) |

---

## Configuration

All tunable pipeline constants live in [`src/utils/config.py`](src/utils/config.py):

| Constant | Default | Description |
|----------|---------|-------------|
| `SPEECH_SIGNAL_RATE_HZ` | `30.0` | Shared timeline sample rate (Hz). Matches video FPS to avoid temporal mismatch. |
| `SPEECH_ENVELOPE_SMOOTH_MS` | `250.0` | Savitzky-Golay smoothing window for the speech envelope (ms). Removes fast phoneme bursts the face landmarker cannot track. |
| `MOUTH_DELTA_WEIGHT` | `0.45` | Blend weight of `|Δmouth|` in the mouth signal. Controls the emphasis on jaw-opening onsets vs absolute mouth position. |
| `MAX_LAG_SECONDS` | `0.4` | NCC search window (±400 ms). Covers expected encoding delays and human reaction drift. |

---

## Output Files

For a job with ID `abc123`, all outputs are isolated under `data/output/abc123/`:

```
data/
├── videos/abc123.mp4
├── audio/abc123.wav
├── frames/abc123/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   └── ...
│   └── video_meta.json          ← {"fps": 30.0, "frame_count": 900}
└── output/abc123/
    ├── mouth_signal.csv         ← frame, time_sec, mouth_height, inner_area,
    │                               velocity, jaw_opening, mouth_signal
    ├── speech_signal.csv        ← time_step, speech_signal
    └── aligned/
        ├── aligned_mouth_signal.csv   ← after lag correction & trim
        ├── aligned_speech_signal.csv  ← after lag correction & trim
        ├── alignment_result.json      ← best_lag_ms, NCC score, n_missing
        ├── correlation_report.json    ← Pearson, cosine, RMSE, lip_sync_score
        ├── signal_overlay_plot.png    ← visual overlay of both signals
        └── decision.json             ← final verdict + warnings
```

---

## Verdict System

The pipeline outputs one of four verdicts:

| Verdict | Score Range | Meaning |
|---------|-------------|---------|
| `SYNCED` | 85–100 | Audio and mouth movement are well correlated — likely genuine |
| `LIKELY_SYNCED` | 65–84 | Good correlation — probably genuine with minor noise |
| `UNCERTAIN` | 45–64 | Weak correlation — could be genuine or dubbed |
| `NOT_SYNCED` | 0–44 | Poor correlation — likely dubbed, deepfake, or pre-recorded |

### Reliability Penalties

On top of the base score, two extra checks can change the result:

- **Suspicious lag penalty (×50%):** If the NCC best lag lands at ≥ 95% of the search window edge (≥ ±380 ms when searching ±400 ms), the algorithm ran off the boundary and found no genuine alignment peak. This is a strong indicator of complete desync. The score is halved before verdict assignment.

- **Missing frames warning:** If more than 30% of mouth frames were interpolated due to failed face detection, the result may be unreliable. A warning is added to the `warnings[]` array in the response.

---

## Running Modules Individually

You can run each pipeline step standalone for debugging or batch processing:

```powershell
# Extract frames from a video
uv run python src/video/frame_extractor.py

# Extract audio track
uv run python src/audio/audio_extractor.py

# Compute mouth-opening features
uv run python src/face/mouth_features.py

# Generate speech signal (mel_band method)
uv run python src/speech/speech_signal.py

# Align mouth and speech signals (NCC)
uv run python src/sync/signal_alignment.py

# Score correlation
uv run python src/sync/correlation.py

# Make final verdict
uv run python src/sync/decision.py
```

> **Windows note:** If you see errors related to `_dispatcher.pyd` (Numba), set this before running:
> ```powershell
> $env:NUMBA_DISABLE_JIT = "1"
> ```
> This is set automatically when running through `main.py` / the API.

---

## Dependencies

| Category | Key Packages |
|----------|-------------|
| Computer Vision | `opencv-python` `mediapipe` `numpy` `scipy` `pillow` |
| Deep Learning | `torch` `torchvision` `torchaudio` |
| Hugging Face | `transformers` `huggingface-hub` `accelerate` |
| Audio | `librosa` `soundfile` `ffmpeg-python` `imageio-ffmpeg` |
| Data Science | `pandas` `scikit-learn` `matplotlib` |
| REST API | `fastapi` `uvicorn[standard]` `python-multipart` |
| Utilities | `tqdm` `rich` `python-dotenv` `pyyaml` |

Full pinned versions are in [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock).

---

## Known Limitations

| Limitation | Impact | Notes |
|------------|--------|-------|
| No face in video | Pipeline fails | Requires a clear, forward-facing speaker throughout |
| Very short clips (< 3 s) | Unreliable score | Not enough signal overlap for meaningful correlation |
| Heavy background noise | Lower speech signal quality | Mel-band is more robust than RMS but not immune |
| Extreme head pose (profile view) | High missing-frame rate | Face landmarker drops detections on strong yaw |
| Whispering | Weak speech signal | Mel-band still outperforms RMS, but signal is faint |
| Multiple simultaneous speakers | Wrong mouth tracked | Tracks the largest/first face found by MediaPipe |
| Blocking `/process` endpoint | ~30 s HTTP timeout | Integrate with a task queue (e.g. Celery/ARQ) for production use |

---

## Logging

Logs are written to two destinations simultaneously:

- **Console** (`INFO` level) — brief format with timestamp and message
- **`logs/lip_sync.log`** (`DEBUG` level) — full format with file path, line number, and function name

To change the log level or silence a specific module, edit [`logging.yaml`](logging.yaml).

---

*Built with MediaPipe · librosa · FastAPI · PyTorch*

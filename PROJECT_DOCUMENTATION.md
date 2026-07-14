# Lip-Sync Mismatch Detection Pipeline — Project Documentation

**Project:** `lip-sync-detection`
**Purpose:** Detect audio-visual desynchronization in video (e.g. video-interview fraud, deepfakes, dubbing errors) by measuring how well mouth movement correlates with speech.
**Stack:** Python 3.12, FastAPI, MediaPipe, Librosa, SciPy/NumPy, PyTorch/Transformers (optional path)

---

## 1. What This Project Does

Given a video file, the pipeline answers one question: **"Does the mouth in this video actually match the speech in the audio track?"**

It does this without any deep-learning classifier trained on "real vs fake" — instead it's a **signal-processing pipeline**: extract a mouth-opening curve from the video, extract a speech-activity curve from the audio, line the two curves up in time, and measure how strongly they move together. A low correlation means the lips are not producing the sounds being heard — a strong signal for lip-sync fraud, bad dubbing, or a deepfake face-swap.

### End-to-end flow

```
Video file
   │
   ├──▶ Phase 1a: Frame extraction (OpenCV)              ─┐
   ├──▶ Phase 1b: Audio extraction (FFmpeg → 16kHz WAV)   ─┤ parallel
   │                                                        │
   ├──▶ Phase 2a: Mouth-opening signal (MediaPipe)        ─┐
   ├──▶ Phase 2b: Speech-activity signal (Mel spectrogram)─┤ parallel
   │
   ├──▶ Phase 3: Signal Alignment (resample, normalize,
   │              smooth, cross-correlate to find lag)
   │
   ├──▶ Phase 4: Correlation scoring (Pearson / cosine / RMSE)
   │
   └──▶ Phase 5: Decision (threshold verdict + reliability checks)
```

The two extraction phases (video-side and audio-side) run **concurrently via `asyncio.gather`**, since they are independent until the alignment step. This is why the pipeline is written with `async`/`await` and `asyncio.to_thread()` throughout — the actual CPU-bound work (OpenCV, MediaPipe, librosa) is not natively async, so it's offloaded to a thread pool while FastAPI's event loop stays free.

---

## 2. Why This Approach — Signal Correlation, Not a Trained Classifier

The most common "state of the art" way to do lip-sync detection is a supervised deep model such as **SyncNet** (Chung & Zisserman) or **Wav2Lip's discriminator**, which are pretrained CNN-based audio-visual embedding networks trained on large lip-sync datasets to output a sync/desync score directly.

This project deliberately does **not** use that approach. Reasons:

| Consideration | Trained sync-classifier (SyncNet-style) | This project's signal-correlation approach |
|---|---|---|
| Training data | Needs a large labeled audio-visual dataset | None required — purely geometric/statistical |
| Interpretability | Black-box score, hard to explain to a stakeholder | Every number (Pearson r, lag in ms, which frames failed) is traceable and explainable in a report |
| Compute cost | Needs GPU inference per short video clip, model download/hosting | Runs on CPU in a few seconds per short clip |
| Domain shift risk | Accuracy drops on video not resembling training distribution (lighting, camera angle, language) | Geometric mouth-opening vs. speech-energy correlation generalizes across languages/cameras since it isn't pattern-matching learned faces |
| Explainability for a fraud-review workflow | Score alone | Verdict + reason + explicit warnings (e.g. "best lag hit the edge of the search window") — better suited to a human reviewer deciding whether to flag an interview |

Given the target use case — **flagging suspicious video interviews for human review**, not fully-automated takedown — an interpretable, dependency-light pipeline was prioritized over a marginally more accurate but opaque deep model. The `SyncDecision` class exists specifically to attach *reasons* to a verdict, which a raw classifier score cannot do on its own.

`src/sync/syncnet.py` exists as a placeholder in the codebase (currently empty) — a true SyncNet-embedding-based scorer was considered as a future accuracy upgrade but was not implemented, for the reasons above plus the added GPU/model-hosting burden it would add to a FastAPI service meant to respond in seconds.

---

## 3. Component-by-Component Breakdown

### 3.1 Face Landmark Detection — MediaPipe `FaceLandmarker` (478-point mesh)

**File:** `src/face/mouth_features.py`, `src/face/face_detector.py`
**Model files bundled:** `models/face_landmarker.task`, `models/blaze_face_short_range.tflite`

**Why MediaPipe:**
- Runs fully on CPU in real time — no GPU dependency for the video side of the pipeline, which matters because this is meant to run inside a synchronous FastAPI request.
- The 478-point face mesh gives dense landmarks around the lips (inner contour + outer contour), which is more than enough resolution to compute a stable mouth-opening signal — competitive face-mesh alternatives with this density either need a GPU (e.g. 3DDFA) or are far heavier to install.
- Ships as a single `.task`/`.tflite` file with no external weight download at runtime, which matters for a project that has been run inside a locked-down corporate environment (see the Windows/Numba note in `main.py` — the machine this is deployed to blocks unsigned DLL execution, so anything requiring compiling native extensions at runtime is a liability).
- Uses the **Tasks API in `VIDEO` running mode**, which does temporal tracking across frames instead of re-detecting the face from scratch every frame — faster and more temporally stable landmarks than running the `IMAGE` mode per frame.

**Alternatives considered and why not used:**

| Alternative | Why not chosen |
|---|---|
| **Dlib 68-point predictor** | Coarser mouth landmarking (only ~20 points around the mouth vs. MediaPipe's fine inner/outer contour), noticeably more jitter frame-to-frame, and dlib's landmark model file is older/less accurate on non-frontal faces. |
| **OpenFace / OpenFace 2.0** | More accurate for facial-action-unit research but is a heavyweight C++ toolkit with a painful build/install process — a poor fit for a service that needs to be `pip`/`uv` installable and deployable without a custom build step. |
| **RetinaFace / MTCNN (as the detector)** | Both are solid face *detectors* but heavier (RetinaFace is CNN-based, needs more compute) than BlazeFace for what's just a "find one face, then run the landmarker" step. BlazeFace (bundled with MediaPipe) is optimized for exactly this lightweight single-face-per-frame use case. |
| **3D Morphable Model fitting (3DDFA, DECA)** | Would give a genuinely 3D jaw-opening measurement, but requires GPU inference and is significantly slower — overkill for a 2D mouth-opening proxy signal, which normalized 2D landmark distances already capture well enough for correlation purposes. |

**What's actually computed** (see `MouthFeatureExtractor._compute_features`): four normalized features per frame, blended into one `mouth_signal`:

1. **`mouth_height`** (weight 0.45) — average of inner-lip gap (landmarks 13↔14) and outer-lip gap (0↔17), normalized by inter-ocular distance (landmarks 33/263) so it's scale-invariant across face size / camera distance.
2. **`inner_area`** (weight 0.20) — shoelace-formula polygon area of the 16-point inner-lip contour, a 2-D corroboration of opening that's less sensitive to jitter on any single landmark pair.
3. **`velocity`** (weight 0.25) — signed frame-to-frame delta of normalized mouth height, computed *after* normalization. This is the only feature carrying timing information (onset/offset) rather than static shape, which is what actually needs to correlate with an audio envelope.
4. **`jaw_opening`** (weight 0.10) — upper-lip-to-chin distance (13→152), capturing jaw drop somewhat independently of pure lip separation.

Frames with no detected face, or a degenerate (near-zero) inter-ocular reference, are recorded as missing and later linearly interpolated rather than treated as "mouth closed" — this was a deliberate fix, since treating a missed detection as a closed mouth would fabricate false desync signal.

### 3.2 Speech Signal Generation — Mel-Band Energy (production default)

**File:** `src/audio/mel_spectrogram.py`, `src/speech/speech_signal.py`

`SpeechSignalGenerator` supports **three interchangeable methods** — this was built as a pluggable interface specifically so the team could A/B the tradeoffs:

| Method | What it measures | Status |
|---|---|---|
| `mel_band` | Summed log-mel power in the **300–3400 Hz speech-articulation band** (telephone-band speech, covering vowel formants F1–F3) | **Production default** (used by `main.py`'s `/process` endpoint) |
| `energy` | Raw RMS amplitude envelope of the waveform | Available, not default |
| `wav2vec_norm` | L2-norm of Wav2Vec2 (`facebook/wav2vec2-base-960h`) hidden-state embeddings | Available, not default |

**Why `mel_band` is the default:**
- RMS energy (`energy` method) measures **loudness**, not articulation — it fails on whispered speech (quiet but full mouth movement) and sustained hums/vowels held at constant volume (loud but the mouth barely changes shape). Mel-band power restricted to the speech-formant range tracks phoneme activity — it rises when the jaw opens for vowels and drops during lip closures (plosives/silence) — which is the physical quantity that should actually correlate with mouth opening.
- It's computed with `librosa.feature.melspectrogram`, which is fast, CPU-only, and has no model weights to download — versus `wav2vec_norm` which requires downloading and running a ~95M-parameter transformer (`Wav2Vec2Model`) through `transformers`/`torch` for every request.

**Why `wav2vec_norm` (Wav2Vec2) exists but isn't the default:**
Wav2Vec2 was evaluated as the "smarter" option — it's a self-supervised speech representation model, and the L2-norm of its hidden states is a reasonable proxy for phonetic activity. It was kept in the codebase (`src/speech/wav2vec_model.py`, `speech_features.py`) as a documented alternative rather than deleted, because:
- It's **much heavier**: loading a transformer + running inference on every uploaded video adds real latency and a large dependency footprint (`torch`, `transformers`, `accelerate`) for a synchronous API endpoint expected to respond in seconds.
- The embedding norm is a **derived scalar of a 768-dim vector** — it's an indirect proxy for "how much is being said," whereas mel-band power in the articulation band is a direct, physically-motivated measurement with no learned black box between the audio and the number.
- Wav2Vec2's native output rate is fixed at 50 Hz (20 ms stride), requiring an extra resampling step to match the pipeline's target rate — one more place a mismatch bug can creep in (see the frequency-band bug fix below).
- No accuracy gain was demonstrated that justified the extra GPU/CPU cost and dependency weight for this use case.

It remains available via `SpeechSignalGenerator(method="wav2vec_norm")` for future experimentation, e.g. if the team later wants to fine-tune a proper phoneme-activity model.

**Bug fix of note:** an earlier version of `mel_spectrogram.py` mapped the 300–3400 Hz articulation band to mel-bin indices incorrectly (hand-derived approximation). It was corrected to use `librosa.mel_frequencies()` directly to resolve the real bin centers for the given `n_mels`/`sample_rate`/`fmax` configuration, so the selected band always matches what `librosa.feature.melspectrogram` actually produced — removing a systematic source of mismatch between the intended and actual frequency band.

### 3.3 Signal Rate Decoupling

**File:** `src/utils/config.py`, `src/sync/signal_alignment.py`

Early iterations generated the speech signal at 100 Hz. This was deliberately reduced to **`SPEECH_SIGNAL_RATE_HZ = 30.0`** (roughly matching typical video FPS) because at 100 Hz the speech signal captured fast phoneme-level transients (plosive bursts, ~5–10 ms) that MediaPipe's landmarker **cannot physically track** at ~33 ms/frame — this produced a systematic mismatch that dragged Pearson correlation toward zero even on perfectly synced video. Decoupling the two signal rates (video-native mouth signal vs. audio-native speech signal, resampled onto one shared timeline in `SignalAlignment.resample()`) and choosing a shared rate close to the video's own frame rate fixed this.

Additional Savitzky–Golay smoothing (`SPEECH_ENVELOPE_SMOOTH_MS = 250.0`) is applied to the speech envelope after normalization for the same reason — matching the effective temporal bandwidth of both signals before cross-correlation.

### 3.4 Robust Normalization

**File:** `src/sync/signal_alignment.py` (`robust_normalize`)

Before comparison, both signals go through: **median filter (spike removal) → percentile clipping (1st/99th) → MinMax scaling to [0, 1]**. A single-frame landmark jump (occlusion, motion blur, a missed detection interpolated badly) would otherwise set a bad max/min and compress the rest of the signal's dynamic range under plain MinMax scaling. This was a targeted accuracy fix — raw MinMax normalization is fragile to exactly the kind of single-frame outliers MediaPipe occasionally produces.

### 3.5 Alignment — FFT-Accelerated Normalized Cross-Correlation (NCC)

**File:** `src/sync/signal_alignment.py` (`cross_correlate`)

Rather than assuming audio/video are already time-aligned, the pipeline searches a window of up to `MAX_LAG_SECONDS = 0.4` s in either direction for the lag that maximizes normalized cross-correlation between the two envelopes, using `scipy.signal.correlate(..., method="fft")` for speed. This matters because:
- Real-world recording setups (webcam + separate mic, containerized video with muxing delay) commonly introduce a fixed audio/video offset.
- A genuine desync video and a merely *offset* video look identical under a naive same-index comparison — searching for the best lag first prevents false positives from an incidental few-hundred-millisecond recording offset.

If the best lag found sits at the very edge of the search window, this is treated as a red flag rather than trusted — `SyncDecision._lag_is_suspicious()` applies a 50% score penalty, since an edge-of-window result means the search never found a genuine correlation peak, which is itself evidence of an unrelated audio/video pairing.

The mouth signal is also blended with its own first derivative (`blend_mouth_with_delta`, weight `MOUTH_DELTA_WEIGHT = 0.45`) before the lag search, since the derivative captures jaw-opening *onsets* — which line up with speech-burst timing far better than absolute lip position does.

### 3.6 Scoring — Pearson Correlation Only

**File:** `src/sync/correlation.py`

The final `lip_sync_score` (0–100) is derived from **Pearson correlation alone**, clamped to `[0, 1] → [0, 100]`. Cosine similarity and RMSE are computed and reported for diagnostic context but deliberately **excluded from scoring** — after MinMax normalization both signals are always non-negative, which pushes cosine similarity to a "free" ~0.5+ floor regardless of whether the signals actually move together in time. Pearson correlation is the only one of the three that specifically measures whether the two curves *rise and fall together*, which is the actual question being asked.

### 3.7 API Layer — FastAPI

**File:** `main.py`

**Why FastAPI over Flask/Django:**
- Native `async def` route support pairs directly with the pipeline's `asyncio.gather`-based parallel phase 1/2 execution — Flask would need extra machinery (e.g. Celery, threads) to get the same concurrency without blocking the worker.
- Automatic request/response validation and OpenAPI docs via Pydantic models (`UploadResponse`, `ProcessResponse`) reduce boilerplate for a small two-endpoint service (`/upload`, `/process/{video_id}`).
- Lighter weight than Django, which brings an ORM, admin panel, and templating system the project doesn't need — this is a processing API, not a web app with views.

---

## 4. Recent Accuracy & Bug-Fix Work

This reflects the debugging/refinement pass most recently done on the pipeline:

1. **Mel-band frequency mapping fix** — corrected the Hz→mel-bin index conversion in `mel_spectrogram.py` to use `librosa.mel_frequencies()` instead of a hand-derived formula, eliminating drift between the intended 300–3400 Hz articulation band and what was actually being summed.
2. **Decoupled signal rates** — separated the speech signal's native generation rate from the video's frame rate (see §3.3), fixing a systematic near-zero correlation bug that affected even correctly synced video.
3. **Robust (outlier-resistant) normalization** — replaced plain MinMax scaling with median-filter + percentile-clip + MinMax (see §3.4), preventing single-frame landmark glitches from compressing the whole signal's dynamic range.
4. **Missing-frame interpolation instead of zero-fill** — frames with no detected face are linearly interpolated from neighbors rather than defaulted to a "closed mouth," avoiding fabricated desync signal from ordinary detection gaps.
5. **Edge-of-window lag detection as a reliability signal** — `SyncDecision` now specifically penalizes results where the NCC search's best lag sits at the boundary of the search window, since that indicates no genuine correlation peak was found at all.

*(Note: sub-sample lag refinement via parabolic interpolation around the NCC peak — a natural next step to get sub-frame lag precision instead of the current whole-sample-step resolution — is not yet implemented in `cross_correlate()`; it's a reasonable near-term improvement, not a currently-shipped feature.)*

---

## 5. Technology Stack Summary

| Layer | Library/Model | Why |
|---|---|---|
| Video I/O | OpenCV (`opencv-python`) | Standard, fast frame decode/encode, no heavier alternative needed |
| Audio extraction | `ffmpeg-python` + `imageio-ffmpeg` | Bundles its own FFmpeg binary — works without a system-wide FFmpeg install, important for locked-down deployment environments |
| Face landmarks | MediaPipe `FaceLandmarker` (478-pt mesh) | CPU-only, bundled model file, dense mouth contour, temporal `VIDEO` mode tracking |
| Face detection (upstream) | MediaPipe BlazeFace (`blaze_face_short_range.tflite`) | Lightweight single-face detector, bundled, no extra download |
| Speech signal | Librosa mel-spectrogram (default) / RMS energy / Wav2Vec2 (optional) | Direct, physically-motivated articulation-band measurement beats loudness (RMS) or an indirect learned embedding norm (Wav2Vec2) for this task, at a fraction of the compute cost |
| Signal processing | SciPy (`interp1d`, `correlate`, `medfilt`, `savgol_filter`), scikit-learn (`MinMaxScaler`) | FFT-accelerated cross-correlation, standard robust-statistics building blocks |
| API | FastAPI + Uvicorn | Native async, automatic validation/docs, lightweight |
| Config | Centralized `src/utils/config.py` constants | Prevents magic numbers drifting out of sync between `pipeline.py`, `signal_alignment.py`, and `decision.py` |

---

## 6. Why Not a Full Deep-Learning Deepfake/Lip-Sync Detector

For completeness, the broader landscape of models this project intentionally did **not** adopt:

- **SyncNet / Wav2Lip discriminator** — pretrained audio-visual sync-embedding CNNs. Rejected: black-box score, GPU dependency, and no in-house labeled data to fine-tune or validate against for this specific use case (video interviews).
- **End-to-end deepfake detectors (e.g. XceptionNet-based classifiers)** — these detect *visual* generation artifacts (blending boundaries, texture inconsistencies), a different problem from *audio-visual* desync. Out of scope for a pipeline specifically about lip-sync correlation.
- **Full 3D face reconstruction (3DDFA/DECA) for jaw angle** — would give a physically truer "mouth opening in 3D," but at GPU-inference cost that isn't justified by the accuracy gain over normalized 2D landmark distances for a correlation-based (not absolute-angle) measurement.

The guiding principle throughout was: **use the simplest, most explainable, CPU-friendly technique that produces a physically meaningful signal**, and keep heavier/learned alternatives (Wav2Vec2, SyncNet-style scoring) as documented, swappable options rather than the default — so the production path stays fast, dependency-light, and auditable for a human reviewer.

---

*Document generated from the current codebase (`src/`, `main.py`, `pyproject.toml`, and pipeline logs) as a technical reference for team-lead/stakeholder review.*
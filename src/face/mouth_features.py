import re
from pathlib import Path
import csv

import cv2
import mediapipe as mp
from mediapipe import tasks
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Aliases for the new Tasks API
FaceLandmarker        = tasks.vision.FaceLandmarker
FaceLandmarkerOptions = tasks.vision.FaceLandmarkerOptions
RunningMode           = tasks.vision.RunningMode
BaseOptions           = tasks.BaseOptions

# Default path to the bundled FaceLandmarker model.
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)

# ---------------------------------------------------------------------------
# MediaPipe 478-point mesh indices for mouth geometry
# (same topology used by FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
# ---------------------------------------------------------------------------

# Single inner-lip pair (kept for mouth_height component)
_INNER_UPPER_LIP = 13
_INNER_LOWER_LIP = 14

# Outer lip midpoints (give a bigger vertical opening signal)
_OUTER_UPPER_LIP = 0
_OUTER_LOWER_LIP = 17

# Outer eye corners — stable cross-expression reference for normalisation
_LEFT_EYE_OUTER  = 33
_RIGHT_EYE_OUTER = 263

# Chin landmark — used with the upper lip center to measure jaw opening
# independently of pure lip separation (feature 4).
_CHIN = 152

# Inner lip contour — traces the actual visible mouth opening, which is a
# much cleaner speech signal than the outer lip boundary (the outer lips
# barely move for many phonemes while the inner opening changes a lot).
_INNER_LIP_CONTOUR = [
    78, 95, 88, 178, 87,
    14,
    317, 402, 318, 324, 308,
    415, 310, 311, 312, 13,
]

# ---------------------------------------------------------------------------
# Composite weights (sum = 1.0). Defined as constants so they can be tuned
# later without touching the composition logic below.
# ---------------------------------------------------------------------------
_W_HEIGHT    = 0.45  # vertical lip gap (inner + outer average) — primary articulation cue
_W_INNERAREA = 0.20  # inner-mouth polygon area — the actual visible opening
_W_VELOCITY  = 0.25  # frame-to-frame change in mouth height — captures speech dynamics/timing
_W_JAW       = 0.10  # jaw displacement — lower-jaw movement independent of lip separation

# Regex used for natural (human) sort of frame filenames.
_DIGIT_SPLIT = re.compile(r"(\d+)")


def _natural_sort_key(path: Path):

    return [
        int(tok) if tok.isdigit() else tok.lower()
        for tok in _DIGIT_SPLIT.split(path.name)
    ]


def _euclidean(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _shoelace_area(pts):
    """Return the unsigned polygon area via the shoelace formula.

    Args:
        pts: list of (x, y) pixel tuples forming a closed polygon.
    Returns:
        Non-negative float area in px².
    """
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


class MouthFeatureExtractor:
    """
    Extract a mouth-opening signal from every video frame using the
    MediaPipe FaceLandmarker Tasks API.

    Per-frame measurements:
      1. mouth_height — average inner + outer vertical lip gap, normalised
                        by inter-ocular distance. Primary articulation cue:
                        tracks how wide the mouth is open at each instant.
      2. inner_area   — polygon area of the 16-pt inner lip contour,
                        normalised by inter-ocular distance squared. A
                        2-D corroboration of the opening that is less
                        sensitive to landmark jitter on any single pair.
      3. velocity     — signed frame-to-frame delta of normalised
                        mouth_height. This is the only feature that carries
                        *timing* information rather than instantaneous
                        shape, which is exactly what correlates with the
                        onset/offset transients in an audio speech signal.
      4. jaw_opening  — distance from the upper lip center (13) to the
                        chin (152), normalised by inter-ocular distance.
                        Captures lower-jaw displacement somewhat
                        independently of pure lip separation (e.g. jaw
                        drop during vowels even when the lips themselves
                        stay relatively parallel).

    These are blended into a single ``mouth_signal`` column consumed by
    downstream signal_alignment.py, which correlates it against an audio
    envelope via Pearson correlation / normalized cross-correlation:

        mouth_signal = (
            _W_HEIGHT    * mouth_height +
            _W_INNERAREA * inner_area +
            _W_VELOCITY  * velocity +
            _W_JAW       * jaw_opening
        )
    """

    def __init__(
        self,
        input_dir: str,
        output_csv: str,
        fps: float = 30.0,
        model_path: str | None = None,
    ):
        self.input_dir  = Path(input_dir)
        self.output_csv = Path(output_csv)
        self.fps        = fps if fps and fps > 0 else 30.0

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        resolved_model = Path(model_path) if model_path else _DEFAULT_MODEL

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {resolved_model}\n"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(resolved_model)
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )

        self.face_landmarker = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

        # ------------------------------------------------------------------
        # Temporal state for the velocity feature. Holds the previous
        # frame's *normalised* mouth_height (not raw pixels), since velocity
        # must be computed after normalisation so it is scale-invariant
        # across shots/cameras just like mouth_height and inner_area.
        # Reset to None whenever the temporal chain is broken (no face /
        # degenerate reference) so we never compute a bogus jump across a
        # detection gap.
        # ------------------------------------------------------------------
        self._prev_mouth_height = None

    def _timestamp_ms(self, frame_idx0: int) -> int:

        ts = int(round((frame_idx0 * 1000.0) / self.fps))
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    # ------------------------------------------------------------------
    # Per-frame geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _px(landmark, w, h):
        """Return pixel (x, y) for a normalised landmark."""
        return landmark.x * w, landmark.y * h

    def _compute_features(self, face, w, h):
        """
        Compute all four normalised mouth features for one face, plus the
        weighted composite mouth_signal. All raw geometry is gathered once
        here and reused across features to avoid duplicated landmark reads.

        Returns
        -------
        dict with keys: mouth_height, inner_area, velocity, jaw_opening,
        mouth_signal — or None if the inter-ocular reference is degenerate
        (near-zero), in which case the frame should be skipped upstream.
        """
        def lm(idx):
            return self._px(face[idx], w, h)

        # ── Inter-ocular reference (shared normaliser for every feature) ──
        lx, ly = lm(_LEFT_EYE_OUTER)
        rx, ry = lm(_RIGHT_EYE_OUTER)
        inter_ocular = _euclidean(lx, ly, rx, ry)

        if inter_ocular < 1e-6:
            return None

        # ── Feature 1 : Mouth height ───────────────────────────────────
        # Average of inner pair (13↔14) and outer pair (0↔17) so
        # the signal is less sensitive to inner-lip noise alone.
        iu_x, iu_y = lm(_INNER_UPPER_LIP)
        il_x, il_y = lm(_INNER_LOWER_LIP)
        ou_x, ou_y = lm(_OUTER_UPPER_LIP)
        ol_x, ol_y = lm(_OUTER_LOWER_LIP)
        inner_gap = _euclidean(iu_x, iu_y, il_x, il_y)
        outer_gap = _euclidean(ou_x, ou_y, ol_x, ol_y)
        mouth_height_px = (inner_gap + outer_gap) / 2.0
        mouth_height = mouth_height_px / inter_ocular

        # ── Feature 2 : Inner lip (mouth opening) area ─────────────────
        inner_pts     = [lm(idx) for idx in _INNER_LIP_CONTOUR]
        inner_area_px2 = _shoelace_area(inner_pts)
        inner_area = inner_area_px2 / (inter_ocular ** 2)

        # ── Feature 3 : Velocity ────────────────────────────────────────
        # Signed frame-to-frame change in *normalised* mouth height.
        # Computed AFTER normalisation so it stays scale-invariant.
        # Sign is preserved on purpose: positive = opening, negative =
        # closing. This is what lets the signal capture articulation
        # transients (onsets/offsets) rather than just static shape,
        # which is the component most directly comparable to the rate
        # of change in an audio amplitude envelope.
        if self._prev_mouth_height is None:
            velocity = 0.0  # first frame — no prior sample to diff against
        else:
            velocity = mouth_height - self._prev_mouth_height

        # Update state for the next frame's velocity computation.
        self._prev_mouth_height = mouth_height

        # ── Feature 4 : Jaw opening ─────────────────────────────────────
        # Distance from upper lip center (13) to chin (152), normalised
        # by inter-ocular distance. Represents lower-jaw displacement
        # somewhat independently of pure lip separation (e.g. a dropped
        # jaw during open vowels even when the lips stay roughly parallel).
        chin_x, chin_y = lm(_CHIN)
        jaw_opening_px = _euclidean(iu_x, iu_y, chin_x, chin_y)
        jaw_opening = jaw_opening_px / inter_ocular

        # ── Composite mouth_signal ──────────────────────────────────────
        mouth_signal = (
            _W_HEIGHT    * mouth_height +
            _W_INNERAREA * inner_area +
            _W_VELOCITY  * velocity +
            _W_JAW       * jaw_opening
        )

        return {
            "mouth_height": mouth_height,
            "inner_area":   inner_area,
            "velocity":     velocity,
            "jaw_opening":  jaw_opening,
            "mouth_signal": mouth_signal,
        }

    def process(self):

        image_paths = sorted(
            self.input_dir.glob("*.jpg"),
            key=_natural_sort_key,
        )

        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.input_dir}"
            )

        with open(self.output_csv, mode="w", newline="") as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow([
                "frame",
                "time_sec",      # frame / fps           — sync key
                "mouth_height",  # normalised vertical opening
                "inner_area",    # normalised inner lip area
                "velocity",      # signed delta of normalised mouth_height
                "jaw_opening",   # normalised upper-lip-to-chin distance
                "mouth_signal",  # weighted composite → downstream
            ])

            logger.info(f"Processing {len(image_paths)} frames...")

            processed        = 0
            skipped_no_face  = 0
            skipped_zero_ref = 0

            for idx, image_path in enumerate(tqdm(
                image_paths,
                desc="  Mouth features",
                unit="fr",
                ncols=80,
                leave=False,
            ), start=1):

                image = cv2.imread(str(image_path))
                rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                frame_idx0   = idx - 1  # idx is 1-based
                timestamp_ms = self._timestamp_ms(frame_idx0)
                time_sec     = frame_idx0 / self.fps

                results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

                # mouth_height, inner_area, velocity, jaw_opening, mouth_signal
                row_values = [None] * 5

                if not results.face_landmarks:
                    skipped_no_face += 1
                    # Detection gap breaks the temporal chain for velocity —
                    # reset so the next valid frame doesn't diff against a
                    # stale value from before the gap.
                    self._prev_mouth_height = None
                else:
                    face = results.face_landmarks[0]
                    h, w, _ = image.shape

                    feats = self._compute_features(face, w, h)

                    if feats is None:
                        skipped_zero_ref += 1
                        # Degenerate reference also breaks the temporal
                        # chain for the same reason as a missing face.
                        self._prev_mouth_height = None
                    else:
                        row_values = [
                            feats["mouth_height"],
                            feats["inner_area"],
                            feats["velocity"],
                            feats["jaw_opening"],
                            feats["mouth_signal"],
                        ]
                        processed += 1

                writer.writerow([idx, f"{time_sec:.6f}", *row_values])

        self.face_landmarker.close()

        logger.info("Finished Mouth Feature Extraction")
        logger.info(f"Frames Processed    : {processed}")
        if skipped_no_face:
            logger.warning(f"Frames Skipped (no face)   : {skipped_no_face}")
        if skipped_zero_ref:
            logger.warning(
                f"Frames Skipped (zero ref)  : {skipped_zero_ref} "
                "(degenerate inter-ocular reference distance)"
            )
        logger.info(f"CSV Saved To        : {self.output_csv}")


if __name__ == "__main__":

    extractor = MouthFeatureExtractor(
        input_dir="data/frames",
        output_csv="data/output/mouth_signal.csv",
        fps=30.0,
    )

    extractor.process()
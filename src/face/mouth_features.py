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
# Composite weights  (sum = 1.0)
# ---------------------------------------------------------------------------
_W_HEIGHT    = 0.70  # vertical lip gap (inner + outer average)
_W_INNERAREA = 0.30   # inner-mouth polygon area — the actual opening

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
                        by inter-ocular distance.
      2. inner_area   — polygon area of the 16-pt inner lip contour,
                        normalised by inter-ocular distance squared.

    These are blended into a single ``mouth_signal`` column consumed by
    downstream signal_alignment.py:

        mouth_signal = 0.70 * mouth_height + 0.30 * inner_area
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
        Compute raw (un-normalised) mouth geometry values for one face.

        Returns
        -------
        dict with keys: inter_ocular_px, mouth_height_px, inner_area_px2
        """
        def lm(idx):
            return self._px(face[idx], w, h)

        # ── Inter-ocular reference ────────────────────────────────────────
        lx, ly = lm(_LEFT_EYE_OUTER)
        rx, ry = lm(_RIGHT_EYE_OUTER)
        inter_ocular = _euclidean(lx, ly, rx, ry)

        # ── Feature 1 : Mouth height ─────────────────────────────────────
        # Average of inner pair (13↔14) and outer pair (0↔17) so
        # the signal is less sensitive to inner-lip noise alone.
        iu_x, iu_y = lm(_INNER_UPPER_LIP)
        il_x, il_y = lm(_INNER_LOWER_LIP)
        ou_x, ou_y = lm(_OUTER_UPPER_LIP)
        ol_x, ol_y = lm(_OUTER_LOWER_LIP)
        inner_gap = _euclidean(iu_x, iu_y, il_x, il_y)
        outer_gap = _euclidean(ou_x, ou_y, ol_x, ol_y)
        mouth_height = (inner_gap + outer_gap) / 2.0

        # ── Feature 2 : Inner lip (mouth opening) area ────────────────────
        inner_pts  = [lm(idx) for idx in _INNER_LIP_CONTOUR]
        inner_area = _shoelace_area(inner_pts)

        return {
            "inter_ocular_px": inter_ocular,
            "mouth_height_px": mouth_height,
            "inner_area_px2":  inner_area,
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
                "mouth_signal",  # weighted composite → downstream
            ])

            logger.info(f"Processing {len(image_paths)} frames...")

            processed        = 0
            skipped_no_face  = 0
            skipped_zero_ref = 0

            for idx, image_path in enumerate(tqdm(image_paths), start=1):

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

                row_values = [None] * 3  # mouth_height, inner_area, mouth_signal

                if not results.face_landmarks:
                    skipped_no_face += 1
                else:
                    face = results.face_landmarks[0]
                    h, w, _ = image.shape

                    feats = self._compute_features(face, w, h)
                    ref   = feats["inter_ocular_px"]

                    if ref < 1e-6:
                        skipped_zero_ref += 1
                    else:
                        mh         = feats["mouth_height_px"] / ref
                        inner_area = feats["inner_area_px2"]  / (ref * ref)

                        composite = _W_HEIGHT * mh + _W_INNERAREA * inner_area

                        row_values = [mh, inner_area, composite]
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
"""
mouth_mesh_visualizer.py

Focused visualizer for the lip-sync mouth signal. Draws ONLY the geometry
behind the four features that feed the final `mouth_signal`:

    1. Mouth Height   -> vertical inner-lip gap (13, 14)
    2. Inner Area     -> filled inner lip contour polygon
    3. Velocity       -> frame-to-frame delta of normalized mouth height
    4. Jaw Opening    -> upper lip (13) to chin (152), inter-ocular normalized

All four are normalized against the inter-ocular distance (33, 263), which
is kept on screen as the normalization reference. The mouth center marker
and inner lip contour outline are also kept as visual anchors.

Everything unrelated to these four features (lip width, MAR, acceleration,
circularity, perimeter, and other experimental metrics) has been removed.

The code is split into three clear stages per frame:
    - _compute_features()   : pure math, no drawing
    - _draw_geometry()      : draws lines/points/polygon on the image
    - _draw_text_overlay()  : renders the left-side text panel

The MediaPipe tracking pipeline / landmark extraction is unchanged.
"""

from pathlib import Path

import cv2
import numpy as np
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

# Default path to the bundled FaceLandmarker model (same convention as the
# other files: models/ dir two levels up from this file's parent).
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Landmark indices — kept identical to MouthFeatureExtractor so the drawn
# geometry always matches the CSV numbers.
# ---------------------------------------------------------------------------
_INNER_UPPER_LIP = 13
_INNER_LOWER_LIP = 14
_CHIN            = 152
_LEFT_EYE_OUTER  = 33
_RIGHT_EYE_OUTER = 263

_INNER_LIP_CONTOUR = [
    78, 95, 88, 178, 87,
    14,
    317, 402, 318, 324, 308,
    415, 310, 311, 312, 13,
]

# Composite mouth_signal weights (mouth_signal computed only from these
# four features, per project spec).
_W_HEIGHT   = 0.45
_W_INNER    = 0.25
_W_VELOCITY = 0.10
_W_JAW      = 0.20

# Velocity color-coding thresholds (in normalized mouth-height units/frame).
_VELOCITY_STATIONARY_THRESH = 0.01

# ---------------------------------------------------------------------------
# Draw styling
# ---------------------------------------------------------------------------
_COLOR_INNER_CONTOUR   = (0, 200, 255)    # orange — inner lip opening polygon
_COLOR_INNER_FILL      = (0, 200, 255)    # orange fill (semi-transparent)
_COLOR_HEIGHT_LINE     = (60, 220, 60)    # green  — mouth height gap
_COLOR_EYE_REF         = (255, 120, 0)    # blue   — inter-ocular reference
_COLOR_JAW_LINE        = (0, 140, 255)    # amber  — jaw opening line
_COLOR_MOUTH_CENTER    = (255, 255, 255)  # white  — mouth center marker
_COLOR_TEXT            = (255, 255, 255)
_COLOR_VEL_OPENING     = (0, 220, 0)      # green
_COLOR_VEL_CLOSING     = (0, 0, 220)      # red
_COLOR_VEL_STATIONARY  = (0, 220, 220)    # yellow

_INNER_FILL_ALPHA = 0.35


def _euclidean(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _shoelace_area(pts):
    n = len(pts)
    area = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


class MouthMeshVisualizer:
    """
    Draws only the geometry behind the four final mouth-signal features
    (mouth height, inner area, velocity, jaw opening) and overlays their
    values in a clean left-side text panel.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_path: str | None = None,
        min_detection_confidence: float = 0.5,
    ):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        resolved_model = Path(model_path) if model_path else _DEFAULT_MODEL

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {resolved_model}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                " -OutFile models/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
        )

        self.face_landmarker = FaceLandmarker.create_from_options(options)

        # Temporal state for velocity: previous frame's normalized mouth
        # height. Reset to None on any detection gap so we never compute a
        # bogus velocity spike across a missing-face gap.
        self._prev_mouth_height = None

    @staticmethod
    def _px(landmark, w, h):
        return landmark.x * w, landmark.y * h

    # -----------------------------------------------------------------
    # Stage 1: FEATURE COMPUTATION (pure math, no drawing)
    # -----------------------------------------------------------------
    def _compute_features(self, face, w, h):
        """
        Computes the four core features from raw landmarks. Returns a dict
        of normalized values plus the raw pixel-space points needed later
        for drawing, or None if the inter-ocular reference is degenerate.
        """

        def lm(idx):
            return self._px(face[idx], w, h)

        # --- normalization reference: inter-ocular distance
        left_eye  = lm(_LEFT_EYE_OUTER)
        right_eye = lm(_RIGHT_EYE_OUTER)
        inter_ocular = _euclidean(*left_eye, *right_eye)

        if inter_ocular < 1e-6:
            return None

        # --- 1. Mouth Height: vertical gap between inner lip landmarks
        upper_lip = lm(_INNER_UPPER_LIP)
        lower_lip = lm(_INNER_LOWER_LIP)
        mouth_height_raw = _euclidean(*upper_lip, *lower_lip)
        mouth_height = mouth_height_raw / inter_ocular

        # --- 2. Inner Mouth Area: shoelace area of inner lip polygon
        inner_pts = [lm(idx) for idx in _INNER_LIP_CONTOUR]
        inner_area_raw = _shoelace_area(inner_pts)
        inner_area = inner_area_raw / (inter_ocular ** 2)

        # --- 3. Velocity: frame-to-frame delta of normalized mouth height
        if self._prev_mouth_height is None:
            velocity = 0.0
        else:
            velocity = mouth_height - self._prev_mouth_height
        # Update state for next frame.
        self._prev_mouth_height = mouth_height

        # --- 4. Jaw Opening: upper lip center (13) to chin (152)
        chin = lm(_CHIN)
        jaw_opening_raw = _euclidean(*upper_lip, *chin)
        jaw_opening = jaw_opening_raw / inter_ocular

        # --- Mouth center marker (visual anchor only, not a signal feature)
        mouth_center = (
            (upper_lip[0] + lower_lip[0]) / 2.0,
            (upper_lip[1] + lower_lip[1]) / 2.0,
        )

        # --- Composite mouth_signal, built only from the four features above
        mouth_signal = (
            _W_HEIGHT * mouth_height
            + _W_INNER * inner_area
            + _W_VELOCITY * velocity
            + _W_JAW * jaw_opening
        )

        return {
            "values": {
                "mouth_height": mouth_height,
                "inner_area": inner_area,
                "velocity": velocity,
                "jaw_opening": jaw_opening,
                "mouth_signal": mouth_signal,
            },
            "points": {
                "left_eye": left_eye,
                "right_eye": right_eye,
                "upper_lip": upper_lip,
                "lower_lip": lower_lip,
                "chin": chin,
                "inner_contour": inner_pts,
                "mouth_center": mouth_center,
            },
        }

    # -----------------------------------------------------------------
    # Stage 2: VISUALIZATION (drawing only, no math)
    # -----------------------------------------------------------------
    @staticmethod
    def _as_int(pt):
        return int(round(pt[0])), int(round(pt[1]))

    def _velocity_color(self, velocity):
        if velocity > _VELOCITY_STATIONARY_THRESH:
            return _COLOR_VEL_OPENING
        elif velocity < -_VELOCITY_STATIONARY_THRESH:
            return _COLOR_VEL_CLOSING
        else:
            return _COLOR_VEL_STATIONARY

    def _draw_geometry(self, image, features):
        pts = features["points"]
        vals = features["values"]
        as_int = self._as_int

        # ---- Eye reference line (kept for normalization visibility)
        cv2.circle(image, as_int(pts["left_eye"]), 3, _COLOR_EYE_REF, -1)
        cv2.circle(image, as_int(pts["right_eye"]), 3, _COLOR_EYE_REF, -1)
        cv2.line(image, as_int(pts["left_eye"]), as_int(pts["right_eye"]), _COLOR_EYE_REF, 1)

        # ---- Inner lip contour: outline + semi-transparent fill (Inner Area)
        contour_int = np.array([as_int(p) for p in pts["inner_contour"]], dtype=np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [contour_int], _COLOR_INNER_FILL)
        cv2.addWeighted(overlay, _INNER_FILL_ALPHA, image, 1 - _INNER_FILL_ALPHA, 0, dst=image)
        cv2.polylines(image, [contour_int], isClosed=True, color=_COLOR_INNER_CONTOUR, thickness=2)

        # ---- Mouth Height: vertical line between inner lip landmarks
        cv2.line(image, as_int(pts["upper_lip"]), as_int(pts["lower_lip"]), _COLOR_HEIGHT_LINE, 2)
        cv2.circle(image, as_int(pts["upper_lip"]), 3, _COLOR_HEIGHT_LINE, -1)
        cv2.circle(image, as_int(pts["lower_lip"]), 3, _COLOR_HEIGHT_LINE, -1)

        # ---- Jaw Opening: upper lip (13) to chin (152)
        cv2.line(image, as_int(pts["upper_lip"]), as_int(pts["chin"]), _COLOR_JAW_LINE, 2)
        cv2.circle(image, as_int(pts["chin"]), 3, _COLOR_JAW_LINE, -1)

        # ---- Mouth center marker (visual anchor)
        cv2.circle(image, as_int(pts["mouth_center"]), 3, _COLOR_MOUTH_CENTER, -1)

        # ---- Velocity: color-coded arrow beside the mouth
        vel = vals["velocity"]
        vel_color = self._velocity_color(vel)
        arrow_origin = (
            int(pts["mouth_center"][0]) + 40,
            int(pts["mouth_center"][1]),
        )
        # Arrow points up when opening, down when closing, flat when stationary.
        arrow_len = 20
        if vel > _VELOCITY_STATIONARY_THRESH:
            arrow_tip = (arrow_origin[0], arrow_origin[1] - arrow_len)
        elif vel < -_VELOCITY_STATIONARY_THRESH:
            arrow_tip = (arrow_origin[0], arrow_origin[1] + arrow_len)
        else:
            arrow_tip = (arrow_origin[0] + arrow_len, arrow_origin[1])
        cv2.arrowedLine(image, arrow_origin, arrow_tip, vel_color, 2, tipLength=0.4)

    # -----------------------------------------------------------------
    # Stage 3: TEXT OVERLAY (left-side panel only)
    # -----------------------------------------------------------------
    def _draw_text_overlay(self, image, features):
        if features is None:
            cv2.putText(
                image, "no face / degenerate reference", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            return

        vals = features["values"]
        vel_color = self._velocity_color(vals["velocity"])

        lines = [
            (f"Mouth Height : {vals['mouth_height']:.4f}", _COLOR_TEXT),
            (f"Inner Area   : {vals['inner_area']:.4f}", _COLOR_TEXT),
            (f"Velocity     : {vals['velocity']:.4f}", vel_color),
            (f"Jaw Opening  : {vals['jaw_opening']:.4f}", _COLOR_TEXT),
            (f"Mouth Signal : {vals['mouth_signal']:.4f}", _COLOR_TEXT),
        ]

        for i, (line, color) in enumerate(lines):
            cv2.putText(
                image, line, (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

    # -----------------------------------------------------------------
    # Per-frame pipeline: compute -> draw geometry -> draw text
    # -----------------------------------------------------------------
    def _process_single_frame(self, image, face_landmarks):
        h, w, _ = image.shape

        features = None
        if face_landmarks:
            features = self._compute_features(face_landmarks, w, h)

        if features is not None:
            self._draw_geometry(image, features)
        else:
            # Detection gap invalidates the temporal chain for velocity.
            self._prev_mouth_height = None

        self._draw_text_overlay(image, features)

    def process_frames(self):
        image_paths = sorted(self.input_dir.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

        logger.info("Processing %d frames from %s", len(image_paths), self.input_dir)

        success_count = 0

        for image_path in tqdm(image_paths):
            image = cv2.imread(str(image_path))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = self.face_landmarker.detect(mp_image)

            face_landmarks = result.face_landmarks[0] if result.face_landmarks else None
            if face_landmarks:
                success_count += 1

            self._process_single_frame(image, face_landmarks)

            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), image)

        self.face_landmarker.close()

        logger.info("Finished mouth mesh visualization")
        logger.info("Frames Processed : %d", len(image_paths))
        logger.info("Face Mesh Found  : %d", success_count)


if __name__ == "__main__":

    visualizer = MouthMeshVisualizer(
        input_dir="data/frames/v3",
        output_dir="data/output/mouth_mesh/v3",
    )

    visualizer.process_frames()
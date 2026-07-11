from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe import tasks
from tqdm import tqdm

# Aliases for the new Tasks API
FaceLandmarker        = tasks.vision.FaceLandmarker
FaceLandmarkerOptions = tasks.vision.FaceLandmarkerOptions
FaceLandmarksConn     = tasks.vision.FaceLandmarksConnections
RunningMode           = tasks.vision.RunningMode
BaseOptions           = tasks.BaseOptions

# Default path to the bundled FaceLandmarker model.
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "face_landmarker.task"
)


class FaceMeshDetector:
    """
    Detect 478 facial landmarks using MediaPipe FaceLandmarker (Tasks API).

    Compatible with mediapipe >= 0.10.21 on Python 3.12+.
    Requires the FaceLandmarker model at `models/face_landmarker.task`.
    Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/
              face_landmarker/float16/latest/face_landmarker.task
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
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
            base_options=BaseOptions(
                model_asset_path=str(resolved_model)
            ),
            running_mode=RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.face_landmarker = FaceLandmarker.create_from_options(options)

    def _draw_landmarks(self, image, face_landmarks):
        """Draw tessellation mesh on image using cv2."""
        h, w, _ = image.shape

        # Convert normalised landmarks → pixel coords
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks
        ]

        # Draw each tessellation edge
        for conn in FaceLandmarksConn.FACE_LANDMARKS_TESSELATION:
            pt1 = pts[conn.start]
            pt2 = pts[conn.end]
            cv2.line(image, pt1, pt2, (80, 110, 10), 1)

    def process_frames(self):

        image_paths = sorted(self.input_dir.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.input_dir}"
            )

        print(f"\nProcessing {len(image_paths)} frames...\n")

        success_count = 0

        for image_path in tqdm(image_paths):

            image = cv2.imread(str(image_path))

            # New Tasks API requires mediapipe.Image (SRGB)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = self.face_landmarker.detect(mp_image)

            if result.face_landmarks:

                success_count += 1

                for face_landmarks in result.face_landmarks:
                    self._draw_landmarks(image, face_landmarks)

            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), image)

        self.face_landmarker.close()

        print("\nFinished")
        print(f"Frames Processed : {len(image_paths)}")
        print(f"Face Mesh Found  : {success_count}")


if __name__ == "__main__":

    detector = FaceMeshDetector(
        input_dir="data/frames",
        output_dir="data/output/face_mesh",
    )

    detector.process_frames()
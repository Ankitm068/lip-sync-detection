from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe import tasks
from tqdm import tqdm

# Aliases for the new Tasks API
MpFaceDetector    = tasks.vision.FaceDetector
FaceDetectorOptions = tasks.vision.FaceDetectorOptions
RunningMode         = tasks.vision.RunningMode
BaseOptions         = tasks.BaseOptions

# Default path to the bundled BlazeFace TFLite model.
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "blaze_face_short_range.tflite"
)


class FaceDetector:
   

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        confidence: float = 0.5,
        model_path: str | None = None,
    ):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        resolved_model = Path(model_path) if model_path else _DEFAULT_MODEL

        if not resolved_model.exists():
            raise FileNotFoundError(
                f"BlazeFace model not found at: {resolved_model}\n"
            )

        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(resolved_model)
            ),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )

        self.face_detector = MpFaceDetector.create_from_options(options)

    def detect_faces(self):

        image_paths = sorted(self.input_dir.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {self.input_dir}"
            )

        print(f"\nProcessing {len(image_paths)} frames...\n")

        detected_faces = 0

        for image_path in tqdm(image_paths):

            image = cv2.imread(str(image_path))

            # New Tasks API requires mediapipe.Image (SRGB)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = self.face_detector.detect(mp_image)

            if result.detections:

                detected_faces += 1

                for detection in result.detections:

                    # New API: bounding_box is in pixels (not relative)
                    bbox = detection.bounding_box

                    xmin   = bbox.origin_x
                    ymin   = bbox.origin_y
                    width  = bbox.width
                    height = bbox.height

                    cv2.rectangle(
                        image,
                        (xmin, ymin),
                        (xmin + width, ymin + height),
                        (0, 255, 0),
                        2,
                    )

            output_path = self.output_dir / image_path.name
            cv2.imwrite(str(output_path), image)

        print("\nFinished")
        print(f"Frames Processed : {len(image_paths)}")
        print(f"Faces Detected   : {detected_faces}")

        self.face_detector.close()


if __name__ == "__main__":

    detector = FaceDetector(
        input_dir="data/frames",
        output_dir="data/output/face_detection",
    )

    detector.detect_faces()
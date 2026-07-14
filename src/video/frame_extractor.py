import json
from pathlib import Path
import cv2
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameExtractor:
    """
    Extracts every frame from a video and saves it as an image.
    Also records the source video's FPS, since downstream signal
    alignment needs it to convert frame indices into real time.
    """

    def __init__(self, video_path: str, output_dir: str):

        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self):

        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError("Unable to open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            raise RuntimeError(
                "Could not read FPS from video; "
                "cannot align signals without it."
            )

        frame_count = 0

        logger.info("Extracting frames from: %s", self.video_path.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        with tqdm(
            total=total_frames,
            desc="  Frames",
            unit="fr",
            ncols=80,
            leave=False,
        ) as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_path = self.output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_count += 1
                pbar.update(1)

        cap.release()

        # Save fps + frame_count alongside the frames so the mouth
        # feature/alignment steps can convert frame index -> seconds.
        meta_path = self.output_dir / "video_meta.json"

        with open(meta_path, "w") as f:
            json.dump(
                {
                    "fps": fps,
                    "frame_count": frame_count,
                },
                f,
                indent=2,
            )

        logger.info("Extraction Complete")
        logger.info(f"Total Frames Saved : {frame_count}")
        logger.info(f"FPS                : {fps:.3f}")
        logger.info(f"Saved To           : {self.output_dir}")
        logger.info(f"Meta Saved To      : {meta_path}")

        return fps, frame_count


if __name__ == "__main__":

    VIDEO_PATH = "data/videos/v3.mp4"

    OUTPUT_DIR = "data/frames/v3/"

    extractor = FrameExtractor(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
    )

    extractor.extract_frames()

    #uv run python src/video/frame_extractor.py
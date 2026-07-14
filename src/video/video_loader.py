from pathlib import Path
import cv2

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoLoader:
    """
    Handles loading a video and retrieving its information.
    """

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open video.")

    def get_video_info(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        return {
            "Filename": self.video_path.name,
            "Resolution": f"{width} x {height}",
            "FPS": round(fps, 2),
            "Total Frames": total_frames,
            "Duration": round(duration, 2),
        }

    def display_video(self):
        logger.info("Starting video preview — press 'Q' to quit.")

        while True:
            success, frame = self.cap.read()

            if not success:
                break

            cv2.imshow("Video Preview", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        self.release()
        cv2.destroyAllWindows()

    def release(self):
        self.cap.release()


if __name__ == "__main__":

    VIDEO_PATH = "data/videos/vid1.mp4"

    loader = VideoLoader(VIDEO_PATH)

    info = loader.get_video_info()

    logger.info("-" * 45)
    logger.info("VIDEO INFORMATION")
    logger.info("-" * 45)

    for key, value in info.items():
        logger.info("%s: %s", key, value)

    logger.info("-" * 45)

    loader.display_video()


    #uv run python src/video/video_loader.py
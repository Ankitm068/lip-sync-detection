from pathlib import Path
import imageio_ffmpeg
import ffmpeg

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioExtractor:
    """
    Extract audio from a video and save it as a WAV file.
    """

    def __init__(self, video_path: str, output_audio: str):
        self.video_path = Path(video_path)
        self.output_audio = Path(output_audio)

        self.output_audio.parent.mkdir(parents=True, exist_ok=True)

    def extract(self):

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        logger.debug("Extracting audio from %s → %s", self.video_path, self.output_audio)

        # Use imageio_ffmpeg's bundled binary — works in any terminal
        # without requiring FFmpeg to be installed system-wide.
        ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()

        (
            ffmpeg
            .input(str(self.video_path))
            .output(
                str(self.output_audio),

                acodec="pcm_s16le",   # 16-bit PCM
                ac=1,                 # Mono
                ar="16000"            # 16 kHz
            )
            .overwrite_output()
            .run(cmd=ffmpeg_cmd, quiet=True)
        )

        logger.info("Audio extracted successfully → %s", self.output_audio)


if __name__ == "__main__":

    extractor = AudioExtractor(
        video_path="data/videos/vid1.mp4",
        output_audio="data/audio/vid1.wav"
    )

    extractor.extract()

    #uv run python src/video/video_loader.py
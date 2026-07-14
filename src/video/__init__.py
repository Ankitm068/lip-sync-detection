"""
src.video — Video Loading & Frame Extraction Sub-package
=========================================================

Exposes all video-related classes so callers can import directly from
the sub-package::

    from src.video import FrameExtractor, VideoLoader

Classes
-------
FrameExtractor
    Reads every frame from a video file using OpenCV and saves each one as
    a numbered JPEG image (``frame_000000.jpg``, …).  Also writes a
    ``video_meta.json`` file containing ``fps`` and ``frame_count`` so
    downstream steps can convert frame indices to real time without
    re-opening the video.

VideoLoader
    Utility class for querying video properties (filename, resolution, FPS,
    frame count, duration) and displaying a live preview window.  Primarily
    used during development; the main pipeline uses ``FrameExtractor``
    directly.
"""

from src.video.frame_extractor import FrameExtractor
from src.video.video_loader import VideoLoader

__all__ = [
    "FrameExtractor",
    "VideoLoader",
]

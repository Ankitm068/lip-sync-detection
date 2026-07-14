"""
src.face — Face Detection & Mouth-Feature Sub-package
======================================================

Exposes all face/mouth-related classes so callers can import directly
from the sub-package::

    from src.face import FaceDetector, MouthFeatureExtractor
    from src.face import MouthMeshVisualizer

Classes
-------
FaceDetector
    Bounding-box face detection using the MediaPipe BlazeFace model
    (``blaze_face_short_range.tflite``).  Used for development visualisation;
    the main pipeline bypasses this and goes straight to FaceLandmarker.

MouthFeatureExtractor
    Core pipeline component.  Runs MediaPipe FaceLandmarker in VIDEO mode
    across all extracted frames, computing four normalised mouth features per
    frame (mouth_height, inner_area, velocity, jaw_opening) and blending them
    into a single ``mouth_signal`` column written to a CSV file.

MouthMeshVisualizer
    Development/debug utility.  Draws the four mouth-feature geometry elements
    (inner lip contour fill, height line, jaw line, velocity arrow) and a
    text overlay directly on each frame image for visual inspection.
"""

from src.face.face_detector import FaceDetector
from src.face.mouth_features import MouthFeatureExtractor
from src.face.face_mesh import MouthMeshVisualizer

__all__ = [
    "FaceDetector",
    "MouthFeatureExtractor",
    "MouthMeshVisualizer",
]

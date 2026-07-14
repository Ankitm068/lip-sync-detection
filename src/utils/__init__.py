"""
src.utils — Shared Utilities Sub-package
=========================================

Exposes logging, file helpers, and pipeline configuration so any module
can import them directly from the sub-package::

    from src.utils import get_logger
    from src.utils import FileUtils
    from src.utils import config

Classes & Functions
-------------------
LoggerFactory
    Owns YAML-based logging configuration.  A singleton instance is created
    at import time; use ``get_logger`` instead of instantiating directly.

get_logger(name)
    Module-level convenience wrapper around ``LoggerFactory.get_logger``.
    Pass ``__name__`` to get a logger whose name mirrors the Python package
    hierarchy (e.g. ``src.audio.audio_loader``).

FileUtils
    Stateless class of ``@staticmethod`` file-system helpers:
    ``ensure_dir``, ``list_frames``, ``read_json``, ``write_json``,
    ``safe_stem``.

config
    Module of shared pipeline constants (``SPEECH_SIGNAL_RATE_HZ``,
    ``SPEECH_ENVELOPE_SMOOTH_MS``, ``MOUTH_DELTA_WEIGHT``,
    ``MAX_LAG_SECONDS``).
"""

from src.utils.logger import LoggerFactory, get_logger
from src.utils.file_utils import FileUtils
from src.utils import config

__all__ = [
    "LoggerFactory",
    "get_logger",
    "FileUtils",
    "config",
]

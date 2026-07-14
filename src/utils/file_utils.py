"""
src/utils/file_utils.py — Common file & directory helpers.

All pipeline steps share these lightweight utilities so path-handling
logic is not duplicated across every module.
"""

import json
import re
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileUtils:
    """
    Stateless collection of file-system helper methods used across the
    lip-sync detection pipeline.

    All methods are ``@staticmethod`` — there is no instance state — so
    you can call them on the class directly::

        FileUtils.ensure_dir("data/output/aligned")
        frames = FileUtils.list_frames("data/frames/job123")
        meta   = FileUtils.read_json("data/frames/job123/video_meta.json")

    Design note
    -----------
    These helpers consolidate the repeated ``Path(...).mkdir(parents=True,
    exist_ok=True)`` and ``json.load`` patterns that used to be scattered
    throughout the pipeline modules, making them easier to test and swap out.
    """

    @staticmethod
    def ensure_dir(path: str | Path) -> Path:
        """
        Create *path* and all missing parents.  No-op if it already exists.

        Parameters
        ----------
        path:
            Directory to create.

        Returns
        -------
        Path
            The resolved ``Path`` object (useful for chaining).
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", p)
        return p

    @staticmethod
    def list_frames(frames_dir: str | Path, pattern: str = "*.jpg") -> list[Path]:
        """
        Return a naturally-sorted list of frame image files.

        Parameters
        ----------
        frames_dir:
            Directory containing the extracted video frames.
        pattern:
            Glob pattern to match (default ``*.jpg``).

        Returns
        -------
        list[Path]
            Sorted list of matching paths.

        Raises
        ------
        FileNotFoundError
            If *frames_dir* does not exist or contains no matching files.
        """
        d = Path(frames_dir)
        if not d.exists():
            raise FileNotFoundError(f"Frames directory not found: {d}")

        paths = sorted(d.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {d}"
            )

        logger.debug("Found %d frame(s) in %s", len(paths), d)
        return paths

    @staticmethod
    def read_json(path: str | Path) -> dict:
        """
        Read and return the contents of a JSON file.

        Parameters
        ----------
        path:
            Path to the ``.json`` file.

        Returns
        -------
        dict
            Parsed JSON object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")

        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)

        logger.debug("JSON loaded from: %s", p)
        return data

    @staticmethod
    def write_json(data: dict, path: str | Path, indent: int = 2) -> Path:
        """
        Serialise *data* to a JSON file, creating parent directories as needed.

        Parameters
        ----------
        data:
            JSON-serialisable dict.
        path:
            Output file path.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        Path
            The path the file was written to.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent)

        logger.debug("JSON written to: %s", p)
        return p

    @staticmethod
    def safe_stem(path: str | Path) -> str:
        """
        Return the file-name stem of *path* with characters that are unsafe
        in directory names replaced by underscores.

        Used to derive deterministic job IDs from video file names.

        Parameters
        ----------
        path:
            Any file path whose stem you want to sanitise.

        Returns
        -------
        str
            Alphanumeric-plus-hyphen stem, safe for use as a directory name.
        """
        stem = Path(path).stem
        return re.sub(r"[^a-zA-Z0-9\-]", "_", stem)

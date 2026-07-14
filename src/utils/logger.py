"""
Logging Configuration Module
=============================

Sets up structured logging for the lip-sync detection pipeline.

The logger is configured via ``logging.yaml`` at the project root when that
file is present.  If it is missing, a safe basicConfig fallback is used so
the application never crashes because of a missing config file.

Key components
--------------
- ``LoggerFactory`` — class that owns setup and hands out named loggers.
- ``get_logger``    — module-level convenience wrapper (backward-compatible).

Format (both console and file)
-------------------------------
    %(asctime)s - %(name)s - %(levelname)s - %(message)s
        - [in %(pathname)s:%(lineno)d  %(funcName)s]

This format includes the **file path, line number, and function name** so
every log entry is immediately linkable to the exact source location that
produced it — critical for debugging a multi-stage ML pipeline.

Log file
--------
    logs/lip_sync.log   (relative to the working directory, auto-created)

Usage
-----
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("detailed diagnostic value: %s", value)
    logger.info("Phase complete")
    logger.warning("Interpolating %d missing frames", n)
    logger.error("Pipeline step failed", exc_info=True)
"""

import logging
import logging.config
import os
import sys
import types
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Numba Mock (Bypass Windows Application Control Policy)
# ---------------------------------------------------------------------------
# librosa 0.11 unconditionally imports numba if it is installed, which fails
# if `_dispatcher.pyd` is blocked by corporate WDAC policies. Because it's an
# optional runtime dependency for our usage, we inject a mock `numba` into
# sys.modules so `librosa.core.audio` succeeds and falls back to pure python.
if "numba" not in sys.modules:
    _mock_numba = types.ModuleType("numba")
    # All decorators should return the original function unmodified
    _mock_numba.jit = lambda *a, **kw: (lambda f: f) if (a or kw) else lambda f: f
    _mock_numba.njit = lambda *a, **kw: (lambda f: f) if (a or kw) else lambda f: f
    _mock_numba.cfunc = lambda *a, **kw: (lambda f: f)
    _mock_numba.stencil = lambda *a, **kw: (lambda f: f)
    _mock_numba.guvectorize = lambda *a, **kw: (lambda f: f)

    _mock_ext = types.ModuleType("numba.extending")
    _mock_ext.overload = lambda *a, **kw: lambda f: f
    _mock_numba.extending = _mock_ext

    sys.modules["numba"] = _mock_numba
    sys.modules["numba.extending"] = _mock_ext

# ---------------------------------------------------------------------------
# Public format string — shared between the YAML fallback and basicConfig
# ---------------------------------------------------------------------------
_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    " - [in %(pathname)s:%(lineno)d %(funcName)s]"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "lip_sync.log"

_YAML_PATH = Path("logging.yaml")


class LoggerFactory:
    """
    Centralised logging factory for the lip-sync detection pipeline.

    Responsibilities
    ----------------
    - Reads ``logging.yaml`` from the project root and calls
      ``logging.config.dictConfig()`` to configure all handlers at once.
    - Falls back to ``logging.basicConfig`` when the YAML file is absent
      so the application always has working log output.
    - Ensures the ``logs/`` directory exists before any FileHandler tries
      to open the log file.
    - Exposes ``get_logger(name)`` to hand out named child loggers to each
      module (the recommended usage pattern throughout the codebase).

    Usage
    -----
    The factory is instantiated once at module import time (see bottom of
    this file). Downstream modules should call the module-level helper::

        from src.utils.logger import get_logger
        logger = get_logger(__name__)

    You can also use the class directly if you need to re-initialise
    logging with a different YAML path (e.g. in tests)::

        factory = LoggerFactory(yaml_path="tests/logging_test.yaml")
        logger  = factory.get_logger("my_test_module")
    """

    def __init__(
        self,
        log_dir: str | Path = _LOG_DIR,
        log_file: str | Path = _LOG_FILE,
        yaml_path: str | Path = _YAML_PATH,
    ) -> None:
        self.log_dir   = Path(log_dir)
        self.log_file  = Path(log_file)
        self.yaml_path = Path(yaml_path)

        self._setup()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """
        Configure the root logger from ``logging.yaml`` (project root).

        Falls back to ``logging.basicConfig`` when the YAML file is absent
        so the application always has working log output.  The log directory
        is created automatically if it does not already exist.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.yaml_path.exists():
            try:
                with open(self.yaml_path, "rt", encoding="utf-8") as fh:
                    config = yaml.safe_load(fh)

                # Patch the file handler's filename at runtime so the path is
                # always resolved relative to the current working directory.
                if (
                    isinstance(config, dict)
                    and "handlers" in config
                    and "file" in config["handlers"]
                ):
                    config["handlers"]["file"]["filename"] = str(self.log_file)

                logging.config.dictConfig(config)
                logging.getLogger(__name__).debug(
                    "Logging configured from %s", self.yaml_path
                )
                return

            except Exception as exc:  # noqa: BLE001
                # Logging isn't working yet — write directly to stderr.
                print(
                    f"[logger.py] WARNING: Could not load {self.yaml_path}: {exc}. "
                    "Falling back to basicConfig.",
                    file=sys.stderr,
                )

        # ------------------------------------------------------------------
        # Fallback: basicConfig with console + file handlers
        # ------------------------------------------------------------------
        logging.basicConfig(
            level=logging.DEBUG,
            format=_LOG_FORMAT,
            datefmt=_DATE_FORMAT,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_file, encoding="utf-8"),
            ],
        )
        logging.getLogger(__name__).warning(
            "logging.yaml not found at '%s'. Using basicConfig fallback.",
            self.yaml_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Return a named logger.

        Parameters
        ----------
        name:
            Typically pass ``__name__`` so the logger hierarchy mirrors the
            Python package structure (e.g. ``src.audio.audio_loader``).

        Returns
        -------
        logging.Logger
        """
        return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Singleton factory — initialised once when this module is first imported.
# ---------------------------------------------------------------------------
_factory = LoggerFactory()


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (keeps all existing call-sites working).
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)

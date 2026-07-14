import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """
    Scores aligned mouth and speech signals for lip-sync quality.

    Pipeline:
        _load_signals()   -> read the aligned mouth/speech CSVs
        compute_metrics()  -> {pearson, cosine, rmse}
        compute_score()     -> (lip_sync_score, verdict)
        print_report()       -> console summary
        save_report()          -> correlation_report.json
        plot_overlay()          -> signal_overlay_plot.png

    Both input signals are the smoothed, normalized mouth/speech
    envelopes produced by SignalAlignment — not derivatives — so a high
    Pearson correlation here means the two envelopes actually rise and
    fall together, which is what the score is meant to measure.
    """

    def __init__(
        self,
        mouth_csv: str,
        speech_csv: str,
        alignment_meta_json: str | None = None,
        output_dir: str | None = None,
    ):
        self.mouth_csv = Path(mouth_csv)
        self.speech_csv = Path(speech_csv)
        self.alignment_meta_json = (
            Path(alignment_meta_json) if alignment_meta_json else None
        )

        # Reuse the alignment step's output folder if we weren't given one.
        if output_dir:
            self.output_dir = Path(output_dir)
        elif self.alignment_meta_json:
            self.output_dir = self.alignment_meta_json.parent
        else:
            self.output_dir = self.mouth_csv.parent

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: load the aligned signals
    # ------------------------------------------------------------------

    def _load_signals(self):
        """Read the aligned mouth and speech signal columns."""
        mouth = pd.read_csv(self.mouth_csv)["mouth_signal"].to_numpy()
        speech = pd.read_csv(self.speech_csv)["speech_signal"].to_numpy()
        return mouth, speech

    def _load_lag_ms(self) -> float:
        """Pull the lag SignalAlignment found, just to include in the report."""
        if self.alignment_meta_json and self.alignment_meta_json.exists():
            with open(self.alignment_meta_json) as f:
                meta = json.load(f)
            return meta.get("best_lag_ms", 0.0)
        return 0.0

    # ------------------------------------------------------------------
    # Step 2: compute metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(mouth: np.ndarray, speech: np.ndarray) -> dict:
        """
        Pearson correlation, cosine similarity, and RMSE between the
        aligned mouth and speech signals. If the signal is too short or
        flat (no variance), correlation is undefined, so we fall back to
        a baseline of 0 instead of raising or returning NaN.
        """
        total_samples = len(mouth)
        std_mouth = np.std(mouth)
        std_speech = np.std(speech)

        if total_samples < 2 or std_mouth < 1e-9 or std_speech < 1e-9:
            warnings.warn(
                "Degenerate or flat signal detected. Forcing correlation "
                "metrics to baseline zero values."
            )
            pearson = 0.0
            cosine = 0.0
            rmse = (
                float(np.sqrt(np.mean((mouth - speech) ** 2)))
                if total_samples > 0
                else 0.0
            )
            return {"pearson": pearson, "cosine": cosine, "rmse": rmse}

        # Pearson: how well the two signals move together (shape match).
        pearson_matrix = np.corrcoef(mouth, speech)
        pearson = float(pearson_matrix[0, 1]) if not np.isnan(pearson_matrix[0, 1]) else 0.0

        # Cosine similarity: angle between the two signal vectors —
        # a secondary shape/magnitude check, not used for scoring.
        norm_m = np.linalg.norm(mouth)
        norm_s = np.linalg.norm(speech)
        cosine = (
            float(np.dot(mouth, speech) / (norm_m * norm_s))
            if (norm_m * norm_s) > 1e-12
            else 0.0
        )

        # RMSE: raw magnitude of the point-by-point difference —
        # reported for context, not used for scoring.
        rmse = float(np.sqrt(np.mean((mouth - speech) ** 2)))

        return {"pearson": pearson, "cosine": cosine, "rmse": rmse}

    # ------------------------------------------------------------------
    # Step 3: compute the lip-sync score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_score(metrics: dict) -> tuple[int, str]:
        """
        Lip-sync score = Pearson correlation only.

        Cosine similarity and RMSE are intentionally excluded from scoring.
        After MinMax normalisation both signals are always in [0, 1], meaning
        they are always non-negative and will produce cosine ~ 0.5+
        regardless of whether they actually correlate in time. Including
        cosine just adds a free ~10-point floor to every video — real
        or fake. Similarly, RMSE only reflects magnitude/offset match, not
        temporal alignment.

        Formula:
          max(0, Pearson)     — primary shape / temporal correlation

        The score is clamped to [0, 1] before converting to a 0-100 scale
        so a negative Pearson cannot push the total below 0. Cosine and RMSE
        are still computed and saved in the report for diagnostic purposes.
        """
        pearson_contrib = max(0.0, metrics["pearson"])

        raw_score = pearson_contrib

        lip_sync_score = int(round(raw_score * 100))
        # Safety clamp — floating-point edge cases.
        lip_sync_score = max(0, min(100, lip_sync_score))

        if lip_sync_score >= 85:
            verdict = "Excellent Sync"
        elif lip_sync_score >= 65:
            verdict = "Good Sync"
        elif lip_sync_score >= 35:
            verdict = "likely Sync"
        else:
            verdict = "Likely Fake / Major Desync"

        return lip_sync_score, verdict

    # ------------------------------------------------------------------
    # Step 4: print + save the report
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(metrics: dict, lag_ms: float, score: int, verdict: str):
        """Log a summary of the correlation results."""
        logger.info("Lip Sync Analysis")
        logger.info("Pearson Correlation : %.2f", metrics['pearson'])
        logger.info("Cosine Similarity   : %.2f", metrics['cosine'])
        logger.info("RMSE                : %.2f", metrics['rmse'])
        logger.info("Temporal Lag        : %+.0f ms", lag_ms)
        logger.info("Lip Sync Score      : %d / 100", score)
        logger.info("Verdict             : %s", verdict)

    def save_report(self, metrics: dict, lag_ms: float, score: int, verdict: str) -> dict:
        """Write the same numbers shown in print_report() to JSON."""
        report_data = {
            "lip_sync_score": score,
            "verdict": verdict,
            "metrics": {
                "pearson": metrics["pearson"],
                "cosine_similarity": metrics["cosine"],
                "rmse": metrics["rmse"],
            },
            "temporal_lag_ms": lag_ms,
        }

        with open(self.output_dir / "correlation_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        return report_data

    # ------------------------------------------------------------------
    # Step 5: plot
    # ------------------------------------------------------------------

    def plot_overlay(self, mouth, speech, verdict, score):
        """Simple time-series overlay of the two aligned signals."""
        plt.figure(figsize=(12, 5))

        plt.plot(mouth, label="Mouth Signal")
        plt.plot(speech, label="Speech Signal")

        plt.title(f"Aligned Signal Overlay — Verdict: {verdict} ({score}/100)")
        plt.xlabel("Shared Timeline Samples")
        plt.ylabel("Normalized Signal Amplitude")

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()

        plot_output_path = self.output_dir / "signal_overlay_plot.png"
        plt.savefig(plot_output_path, dpi=200)
        plt.close()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        plt.switch_backend("Agg")  # headless rendering, no UI thread

        mouth, speech = self._load_signals()
        metrics = self.compute_metrics(mouth, speech)
        lag_ms = self._load_lag_ms()
        score, verdict = self.compute_score(metrics)

        self.print_report(metrics, lag_ms, score, verdict)
        report_data = self.save_report(metrics, lag_ms, score, verdict)
        self.plot_overlay(mouth, speech, verdict, score)

        return report_data


if __name__ == "__main__":
    analyzer = CorrelationAnalyzer(
        mouth_csv="data/output/aligned/aligned_mouth_signal.csv",
        speech_csv="data/output/aligned/aligned_speech_signal.csv",
        alignment_meta_json="data/output/aligned/alignment_result.json",
    )
    analyzer.compute()
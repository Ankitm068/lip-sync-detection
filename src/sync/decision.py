import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SyncDecision:
    """
    Converts the final Lip Sync Score (0-100, from CorrelationAnalyzer)
    into a simple synchronization verdict, while separately checking
    whether the alignment step behind that score can actually be
    trusted.

    Score classification is deliberately simple threshold logic — all
    the real signal-processing judgment (Pearson, cosine, RMSE) already
    happened in CorrelationAnalyzer. This class's only extra job is
    flagging cases where the *alignment* looks shaky (e.g. the best lag
    sat at the edge of the search window, or too many mouth frames were
    interpolated), since a high score built on a bad alignment isn't
    trustworthy.
    """

    def __init__(
        self,
        lip_sync_score: int,
        alignment_meta: dict | None = None,
        correlation_metrics: dict | None = None,
        max_lag_seconds: float = 1.0,
    ):
        self.lip_sync_score = lip_sync_score
        self.alignment_meta = alignment_meta or {}
        self.correlation_metrics = correlation_metrics or {}
        self.max_lag_seconds = max_lag_seconds

    # ------------------------------------------------------------------
    # Reliability checks (unchanged from the correlation-based version)
    # ------------------------------------------------------------------

    def _lag_is_suspicious(self):
        """
        If the best lag found sits right at the edge of the search
        window, the alignment step may just be reporting whatever the
        window happened to include, not a genuine sync offset -- worth
        flagging rather than trusting blindly.
        """
        lag_ms = self.alignment_meta.get("best_lag_ms")

        if lag_ms is None:
            return False

        edge_ms = self.max_lag_seconds * 1000

        return abs(lag_ms) >= edge_ms * 0.95

    def _too_many_missing_frames(self):
        """More than 30% interpolated mouth frames means a big chunk of
        the signal was guessed, not measured -- flag it as a caveat."""
        n_missing = self.alignment_meta.get("n_missing_mouth_frames", 0)
        aligned_len = self.alignment_meta.get("aligned_length")

        return bool(
            aligned_len and n_missing and (n_missing / max(aligned_len, 1)) > 0.3
        )

    # ------------------------------------------------------------------
    # Score -> verdict
    # ------------------------------------------------------------------

    def classify(self):
        """Map the Lip Sync Score onto a verdict, then attach any
        alignment-reliability warnings on top of it."""
        score = self.lip_sync_score
        warnings = []

        if score is None:
            return {
                "lip_sync_score": score,
                "verdict": "UNKNOWN",
                "reason": "Lip Sync Score could not be computed.",
                "warnings": warnings,
            }

        # ── Reliability penalty: suspicious lag ───────────────────────
        # If the NCC search found its best lag at the very edge of the
        # search window (±max_lag_seconds), it means the algorithm never
        # found a genuine alignment peak — it just ran off the boundary.
        # This is the strongest single indicator of a deepfake or a
        # completely unsynchronised track.
        #
        # Applying a 50% score penalty HERE (before verdict assignment)
        # means this structural failure actually changes the outcome
        # rather than just appearing as a footnote warning.
        if self._lag_is_suspicious():
            lag_ms = self.alignment_meta.get("best_lag_ms", 0.0)
            penalty_pct = 50
            score = int(round(score * (1 - penalty_pct / 100)))
            warnings.append(
                f"Best lag ({lag_ms:+.1f} ms) hit the edge of the "
                f"search window (±{self.max_lag_seconds * 1000:.0f} ms). "
                "No genuine alignment peak was found — the NCC search "
                "ran off the boundary, which strongly indicates the audio "
                "and video tracks are not correlated. Score penalised by "
                f"{penalty_pct}%."
            )

        # ── Reliability warning: too many missing frames ───────────────
        if self._too_many_missing_frames():
            n_missing = self.alignment_meta.get("n_missing_mouth_frames", 0)
            warnings.append(
                f"{n_missing} mouth frames were interpolated due to "
                "missed face detections — result may be unreliable."
            )

        # ── Verdict from (possibly penalised) score ────────────────────
        if score >= 85:
            verdict = "SYNCED"
            reason = "Lip Sync Score indicates excellent synchronization."
        elif score >= 65:
            verdict = "LIKELY_SYNCED"
            reason = "Lip Sync Score indicates good synchronization."
        elif score >= 45:
            verdict = "UNCERTAIN"
            reason = "Lip Sync Score indicates possible desynchronization."
        else:
            # Check for uncertainty in low scores
            lag_ms = self.alignment_meta.get("best_lag_ms", 0.0)
            cosine = self.correlation_metrics.get("cosine_similarity", 0.0)
            
            if abs(lag_ms) <= 100 and cosine >= 0.70:
                verdict = "UNCERTAIN"
                reason = "Lip Sync Score is low, but high cosine similarity and small temporal lag suggest uncertainty rather than definitive desynchronization."
            else:
                verdict = "NOT_SYNCED"
                reason = "Lip Sync Score indicates major desynchronization."

        return {
            "lip_sync_score": score,
            "verdict": verdict,
            "reason": reason,
            "warnings": warnings,
        }

    def report(self):
        """Log the verdict + warnings, and return the same dict for saving."""
        result = self.classify()

        logger.info("Sync Decision")
        logger.info("Lip Sync Score : %s", result['lip_sync_score'])
        logger.info("Verdict        : %s", result['verdict'])
        logger.info("Reason         : %s", result['reason'])

        if result["warnings"]:
            for w in result["warnings"]:
                logger.warning("Decision warning: %s", w)

        return result


if __name__ == "__main__":

    # Lip Sync Score + verdict come from CorrelationAnalyzer's report...
    with open("data/output/aligned/correlation_report.json") as f:
        correlation_report = json.load(f)

    # ...but lag/interpolation info for the reliability checks still
    # comes from SignalAlignment's own report.
    with open("data/output/aligned/alignment_result.json") as f:
        alignment_meta = json.load(f)

    decision = SyncDecision(
        lip_sync_score=correlation_report.get("lip_sync_score"),
        alignment_meta=alignment_meta,
        correlation_metrics=correlation_report.get("metrics"),
        max_lag_seconds=1.0,
    )

    result = decision.report()

    output_path = Path("data/output/aligned/decision.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Decision Saved To : %s", output_path)
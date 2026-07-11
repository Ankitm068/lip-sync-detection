import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import correlate, medfilt, savgol_filter
from sklearn.preprocessing import MinMaxScaler


class SignalAlignment:
    """
    Aligns a per-frame mouth-opening signal with a speech-envelope signal
    for lip-sync detection, using FFT-accelerated Normalized
    Cross-Correlation (NCC) to find the best lag.

    Pipeline: load_data -> handle_missing_frames -> resample -> normalize
    -> smooth -> cross_correlate -> apply_lag -> save_results.

    NOTE: cross-correlation runs on the same smoothed, raw (non-derivative)
    envelope signals that get saved as the aligned output, so whatever lag
    is found is the lag downstream Pearson-correlation scoring actually
    sees — no mismatch between the signal aligned against and the signal
    scored.
    """

    def __init__(
        self,
        mouth_csv: str,
        speech_csv: str,
        output_dir: str,
        video_fps: float | None = None,
        video_meta_json: str | None = None,
        speech_rate_hz: float = 50.0,
        max_lag_seconds: float = 1.0,
        smoothing_window_seconds: float = 0.22,
        speech_envelope_smooth_ms: float = 250.0,
        mouth_delta_weight: float = 0.35,
    ):
        self.mouth_csv = Path(mouth_csv)
        self.speech_csv = Path(speech_csv)
        self.output_dir = Path(output_dir)
        self.speech_rate_hz = speech_rate_hz
        self.max_lag_seconds = max_lag_seconds
        self.smoothing_window_seconds = smoothing_window_seconds
        # Additional speech-envelope smoothing (ms) to remove fast phoneme
        # oscillations the face landmarker can't physically track.
        self.speech_envelope_smooth_ms = speech_envelope_smooth_ms
        # Weight of the mouth delta (first derivative) in the blended
        # mouth signal — captures jaw-opening onsets rather than absolute
        # lip position.
        self.mouth_delta_weight = mouth_delta_weight

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FPS can be passed directly, or read from the video_meta.json
        # that FrameExtractor writes out.
        if video_fps is not None:
            self.video_fps = video_fps
        elif video_meta_json is not None:
            with open(video_meta_json) as f:
                meta = json.load(f)
            self.video_fps = meta["fps"]
        else:
            raise ValueError(
                "Must provide either video_fps or video_meta_json "
                "so mouth-signal frame indices can be converted to "
                "real time."
            )

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------

    def load_data(self):
        """Read the mouth and speech CSVs into raw numpy arrays."""
        mouth_df = pd.read_csv(self.mouth_csv)
        speech_df = pd.read_csv(self.speech_csv)

        mouth_signal = mouth_df["mouth_signal"].to_numpy()
        # Missing speech samples default to 0 (silence) rather than being
        # dropped, so the time axis stays evenly spaced.
        speech_signal = speech_df["speech_signal"].fillna(0).to_numpy()

        return mouth_signal, speech_signal

    # ------------------------------------------------------------------
    # 2. Handle missing mouth frames
    # ------------------------------------------------------------------

    def handle_missing_frames(self, mouth_signal):
        """Linearly interpolate frames with no detected face (NaN) from
        their neighbours — treating them as 0 would fake a mouth-closing."""
        missing = pd.isna(mouth_signal)
        n_missing = int(missing.sum())

        if missing.all():
            raise ValueError(
                "No frames had a detected mouth opening; cannot "
                "build a mouth signal."
            )

        if n_missing:
            print(
                f"\nWarning: {n_missing}/{len(mouth_signal)} frames "
                "had no face/mouth detected. Interpolating across "
                "gaps instead of treating them as a closed mouth."
            )
            idx = np.arange(len(mouth_signal))
            mouth_signal = np.interp(idx, idx[~missing], mouth_signal[~missing])

        return mouth_signal, n_missing

    # ------------------------------------------------------------------
    # 3. Resample onto a shared time axis
    # ------------------------------------------------------------------

    def resample(self, mouth_signal, speech_signal):
        """Mouth (video_fps) and speech (speech_rate_hz) run at different
        rates — linearly interpolate both onto a shared time axis at
        speech_rate_hz so later steps compare like-for-like samples."""
        mouth_duration = (
            (len(mouth_signal) - 1) / self.video_fps if len(mouth_signal) > 1 else 0.0
        )
        speech_duration = (
            (len(speech_signal) - 1) / self.speech_rate_hz if len(speech_signal) > 1 else 0.0
        )

        mouth_times = np.arange(len(mouth_signal)) / self.video_fps
        speech_times = np.arange(len(speech_signal)) / self.speech_rate_hz

        common_duration = min(mouth_duration, speech_duration)
        n_shared = max(int(round(common_duration * self.speech_rate_hz)) + 1, 2)
        shared_times = np.linspace(0, common_duration, n_shared)

        mouth_interp = interp1d(
            mouth_times,
            mouth_signal,
            kind="linear",
            bounds_error=False,
            fill_value=(mouth_signal[0], mouth_signal[-1]),
        )
        speech_interp = interp1d(
            speech_times,
            speech_signal,
            kind="linear",
            bounds_error=False,
            fill_value=(speech_signal[0], speech_signal[-1]),
        )

        mouth_resampled = mouth_interp(shared_times)
        speech_resampled = speech_interp(shared_times)

        print(
            f"\nResampled both signals to {len(shared_times)} samples "
            f"@ {self.speech_rate_hz:.3f} Hz"
        )

        return mouth_resampled, speech_resampled

    # ------------------------------------------------------------------
    # 4. Normalize
    # ------------------------------------------------------------------

    @staticmethod
    def robust_normalize(signal, median_kernel=5, clip_percentiles=(1, 99)):
        """Normalize to [0, 1] while resisting single-frame outliers (e.g.
        a landmark jump from occlusion/motion blur) that would otherwise
        set a bad max/min and compress the rest of the signal's range.
        Median-filter spikes -> percentile-clip remaining extremes -> MinMax."""
        signal = np.asarray(signal, dtype=float)

        if median_kernel and median_kernel > 1 and len(signal) > median_kernel:
            kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
            signal = medfilt(signal, kernel_size=kernel)

        lo_pct, hi_pct = clip_percentiles
        lo, hi = np.percentile(signal, [lo_pct, hi_pct])
        if hi - lo < 1e-9:
            # Degenerate (near-constant) signal — fall back to raw min/max
            # so we don't clip everything to a single value.
            lo, hi = signal.min(), signal.max()
        signal = np.clip(signal, lo, hi)

        scaler = MinMaxScaler()
        signal = scaler.fit_transform(signal.reshape(-1, 1))
        return signal.flatten()

    # ------------------------------------------------------------------
    # 5. Smooth
    # ------------------------------------------------------------------

    def smooth(self, signal):
        """Light Savitzky-Golay smoothing (NOT a derivative) to remove
        residual landmark jitter while preserving the signal's shape."""
        window = int(round(self.smoothing_window_seconds * self.speech_rate_hz))
        if window % 2 == 0:
            window += 1
        window = min(window, len(signal) - (1 - len(signal) % 2))
        window = max(window, 5)  # must be > polyorder

        return savgol_filter(signal, window_length=window, polyorder=3)

    def smooth_speech_envelope(self, speech_signal):
        """Apply a wider Savitzky-Golay pass to the speech envelope
        after normalization to remove residual fast-phoneme oscillations
        that the face landmarker physically cannot track at video FPS.

        This is separate from the standard `smooth()` call (which is
        applied to the mouth signal for jitter removal) because the
        required window is larger — ~250 ms vs ~220 ms — and it is only
        needed for the speech side.
        """
        if self.speech_envelope_smooth_ms <= 0:
            return speech_signal

        window = int(round((self.speech_envelope_smooth_ms / 1000.0) * self.speech_rate_hz))
        if window % 2 == 0:
            window += 1
        window = min(window, len(speech_signal) - (1 - len(speech_signal) % 2))
        window = max(window, 5)  # must be > polyorder

        # Use polyorder=1 to make this a true moving average filter.
        # A cubic polynomial (polyorder=3) preserves sharp peaks (fast phonemes),
        # leaving the speech signal too jagged to match a sluggish human jaw.
        return savgol_filter(speech_signal, window_length=window, polyorder=1)

    def blend_mouth_with_delta(self, mouth_signal):
        """Blend the absolute mouth-opening signal with its first derivative.

        The blended signal combines:
          - Absolute position  (1 - weight): slow envelope capturing
            overall mouth openness — good for sustained vowels.
          - Delta / onset signal  (weight):  first derivative, capturing
            jaw-opening *onsets* and *offsets* — these align tightly with
            consonant-vowel transitions and speech burst timing.

        Both components are individually normalized to [0, 1] before
        blending so that neither dominates by scale.
        """
        if self.mouth_delta_weight <= 0:
            return mouth_signal

        # First derivative via central differences.
        delta = np.gradient(mouth_signal)
        # Delta has both positive (opening) and negative (closing) values;
        # take the absolute value so both directions of mouth movement
        # contribute a positive signal.
        delta_abs = np.abs(delta)

        # Normalize delta to [0, 1] independently.
        delta_min, delta_max = delta_abs.min(), delta_abs.max()
        if delta_max - delta_min < 1e-9:
            # Degenerate (no mouth movement detected) — skip blending.
            return mouth_signal
        delta_norm = (delta_abs - delta_min) / (delta_max - delta_min)

        w = self.mouth_delta_weight
        blended = (1.0 - w) * mouth_signal + w * delta_norm

        # Re-normalize the blend to [0, 1] so downstream steps see a
        # consistent scale regardless of the weight chosen.
        b_min, b_max = blended.min(), blended.max()
        if b_max - b_min < 1e-9:
            return blended
        return (blended - b_min) / (b_max - b_min)

    # ------------------------------------------------------------------
    # 6. Cross-correlation (NCC, FFT-accelerated)
    # ------------------------------------------------------------------

    @staticmethod
    def cross_correlate(a, b, max_lag_samples, min_overlap=10):
        """Slide `b` (speech) against `a` (mouth) over +/- max_lag_samples
        and return (best_lag, best_corr) via FFT-accelerated NCC. Positive
        lag = speech trails the video; negative = speech leads it."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        n_a, n_b = len(a), len(b)

        a_std, b_std = a.std(), b.std()
        if a_std < 1e-9 or b_std < 1e-9 or n_a < min_overlap or n_b < min_overlap:
            # Degenerate/too-short signal — no meaningful lag to find.
            return 0, float("nan")

        a_norm = (a - a.mean()) / a_std
        b_norm = (b - b.mean()) / b_std

        full = correlate(a_norm, b_norm, mode="full", method="fft")
        k = np.arange(len(full))
        lag_of_k = (n_b - 1) - k

        # Overlap length (number of summed terms) at each lag.
        overlap = np.minimum.reduce(
            [
                np.full_like(k, n_a),
                np.full_like(k, n_b),
                k + 1,
                (n_a + n_b - 1) - k,
            ]
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            ncc = full / overlap

        mask = (lag_of_k >= -max_lag_samples) & (lag_of_k <= max_lag_samples)
        lags = lag_of_k[mask]
        corrs = ncc[mask]
        ovl = overlap[mask]
        corrs = np.where(ovl < min_overlap, np.nan, corrs)
        # NCC on globally-normalized signals is bounded close to [-1, 1]
        # but not guaranteed exact at short overlaps; clip defensively.
        corrs = np.clip(corrs, -1.0, 1.0)

        if np.all(np.isnan(corrs)):
            return 0, float("nan")

        best_idx = int(np.nanargmax(corrs))
        return int(lags[best_idx]), float(corrs[best_idx])

    # ------------------------------------------------------------------
    # 7. Apply the lag
    # ------------------------------------------------------------------

    @staticmethod
    def apply_lag(mouth_signal, speech_signal, best_lag):
        """Shift the two signals by `best_lag` samples and trim to the
        overlapping region so both arrays end up the same length."""
        if best_lag < 0:
            mouth_aligned = mouth_signal[-best_lag:]
            speech_aligned = speech_signal[: len(mouth_aligned)]
        elif best_lag > 0:
            speech_aligned = speech_signal[best_lag:]
            mouth_aligned = mouth_signal[: len(speech_aligned)]
        else:
            n = min(len(mouth_signal), len(speech_signal))
            mouth_aligned = mouth_signal[:n]
            speech_aligned = speech_signal[:n]

        n = min(len(mouth_aligned), len(speech_aligned))
        return mouth_aligned[:n], speech_aligned[:n]

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------

    def save_results(self, mouth_aligned, speech_aligned, best_lag, best_corr, n_missing):
        pd.DataFrame({"mouth_signal": mouth_aligned}).to_csv(
            self.output_dir / "aligned_mouth_signal.csv", index=False
        )
        pd.DataFrame({"speech_signal": speech_aligned}).to_csv(
            self.output_dir / "aligned_speech_signal.csv", index=False
        )

        best_lag_ms = (best_lag / self.speech_rate_hz) * 1000

        with open(self.output_dir / "alignment_result.json", "w") as f:
            json.dump(
                {
                    "best_lag_ms": best_lag_ms,
                    "correlation_score": None if np.isnan(best_corr) else float(best_corr),
                    "speech_rate_hz": self.speech_rate_hz,
                    "max_lag_seconds": self.max_lag_seconds,
                    "aligned_length": len(mouth_aligned),
                    "n_missing_mouth_frames": n_missing,
                },
                f,
                indent=2,
            )

        return best_lag_ms

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self):
        # 1. Load
        mouth_signal, speech_signal = self.load_data()
        print("\nOriginal Signal Lengths")
        print("-------------------------")
        print(f"Mouth  : {len(mouth_signal)} frames @ {self.video_fps:.3f} fps")
        print(f"Speech : {len(speech_signal)} frames @ {self.speech_rate_hz:.3f} Hz")

        # 2. Handle missing mouth frames
        mouth_signal, n_missing = self.handle_missing_frames(mouth_signal)

        # 3. Resample onto a shared time axis
        mouth_signal, speech_signal = self.resample(mouth_signal, speech_signal)

        # 4. Normalize (robust, outlier-resistant)
        mouth_signal = self.robust_normalize(mouth_signal)
        speech_signal = self.robust_normalize(speech_signal)

        # 5a. Smooth mouth signal — light jitter removal, not a derivative.
        mouth_signal = self.smooth(mouth_signal)

        # 5b. Smooth speech envelope — wider window to remove fast phoneme
        #     oscillations the face landmarker physically cannot track.
        speech_signal = self.smooth_speech_envelope(speech_signal)

        # 5c. Blend mouth signal with its first derivative (onset signal).
        #     The delta component captures jaw-opening onsets that align
        #     better with speech burst timing than absolute position alone.
        mouth_signal = self.blend_mouth_with_delta(mouth_signal)

        print(f"\nSignal Processing")
        print("-------------------------")
        print(f"Speech envelope smooth : {self.speech_envelope_smooth_ms:.0f} ms window")
        print(f"Mouth delta blend      : {self.mouth_delta_weight:.0%} delta weight")

        # 6. Cross-correlate to find the best lag
        max_lag_samples = int(round(self.max_lag_seconds * self.speech_rate_hz))
        best_lag, best_corr = self.cross_correlate(
            mouth_signal, speech_signal, max_lag_samples
        )

        print("\nLag Search (NCC)")
        print("-------------------------")
        print(f"Searched         : +/-{self.max_lag_seconds * 1000:.0f} ms")
        print(f"Best Lag         : {(best_lag / self.speech_rate_hz) * 1000:+.1f} ms")
        print(f"Best Correlation : {best_corr:.4f}")
        print(
            "(positive lag = speech trails the mouth movement; "
            "negative lag = speech leads it)"
        )

        # 7. Apply the lag to the same smoothed, blended signals used
        #    for the correlation search so the saved output matches what
        #    was actually aligned.
        mouth_aligned, speech_aligned = self.apply_lag(mouth_signal, speech_signal, best_lag)

        # 8. Save results
        best_lag_ms = self.save_results(
            mouth_aligned, speech_aligned, best_lag, best_corr, n_missing
        )

        print("\nAlignment Complete")
        print("-------------------------")
        print(f"Aligned Length : {len(mouth_aligned)}")

        return mouth_aligned, speech_aligned, best_lag_ms


if __name__ == "__main__":
    alignment = SignalAlignment(
        mouth_csv="data/output/mouth_signal.csv",
        speech_csv="data/output/speech_signal.csv",
        output_dir="data/output/aligned",
        video_meta_json="data/frames/video_meta.json",
        speech_rate_hz=50.0,
        max_lag_seconds=0.4,
    )
    alignment.process()
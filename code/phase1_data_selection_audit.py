"""Phase 1 (Data Selection) — Reproducible data audit + initial feature engineering.

This script:
1) Validates the local EEG dataset files.
2) Extracts window-level EEG features into a tidy tabular dataset.
3) Produces required Phase 1 summaries: missingness, stats, duplicates/outliers, example rows, and plots.

Important note about your workspace:
- Your .vhdr and .vmrk files currently appear to be HTML pages (likely saved from GIN web UI),
  not BrainVision header/marker files. When that happens, MNE cannot load the BrainVision dataset.
- This script will detect that and fall back to a simple binary reader using configuration assumptions.

Run:
  python code/phase1_data_selection_audit.py

Outputs:
  outputs/phase1_data_selection/
    features.csv
    missingness.csv
    summary_stats.csv
    duplicates.csv
    outliers_summary.csv
    example_rows.csv
    plot_histograms.png
    plot_corr_heatmap.png
    plot_windows_per_participant.png

All comments are in English per project rule.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib/seaborn are used only for saving plots.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import welch
from scipy.stats import kurtosis, skew


# -----------------------------------------------------------------------------
# Configuration import (with safe fallbacks)
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("EEG data")
DEFAULT_PARTICIPANTS = ["P2", "P3", "P4"]
DEFAULT_SAMPLING_RATE = 500
DEFAULT_WINDOW_SIZE = 1000
DEFAULT_WINDOW_OVERLAP = 0.5
DEFAULT_NUM_CHANNELS = 64
DEFAULT_DATA_TYPE = "int16"
DEFAULT_MAX_WINDOWS_PER_PARTICIPANT = 300
DEFAULT_AUTO_INFER_LAYOUT = True
DEFAULT_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 100),
}

try:
    # Allow running as `python code/phase1_data_selection_audit.py`.
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    import config as project_config

    DATA_DIR = Path(getattr(project_config, "DATA_DIR", str(DEFAULT_DATA_DIR)))
    PARTICIPANTS = list(getattr(project_config, "PARTICIPANTS", DEFAULT_PARTICIPANTS))
    SAMPLING_RATE = int(getattr(project_config, "SAMPLING_RATE", DEFAULT_SAMPLING_RATE))
    WINDOW_SIZE = int(getattr(project_config, "WINDOW_SIZE", DEFAULT_WINDOW_SIZE))
    WINDOW_OVERLAP = float(getattr(project_config, "WINDOW_OVERLAP", DEFAULT_WINDOW_OVERLAP))
    NUM_CHANNELS = int(getattr(project_config, "NUM_CHANNELS", DEFAULT_NUM_CHANNELS))
    DATA_TYPE = str(getattr(project_config, "DATA_TYPE", DEFAULT_DATA_TYPE))
    MAX_WINDOWS_PER_PARTICIPANT = int(
        getattr(project_config, "MAX_WINDOWS_PER_PARTICIPANT", DEFAULT_MAX_WINDOWS_PER_PARTICIPANT)
    )
    AUTO_INFER_LAYOUT = bool(getattr(project_config, "AUTO_INFER_LAYOUT", DEFAULT_AUTO_INFER_LAYOUT))
    FREQUENCY_BANDS = dict(getattr(project_config, "FREQUENCY_BANDS", DEFAULT_BANDS))
except Exception:
    DATA_DIR = DEFAULT_DATA_DIR
    PARTICIPANTS = DEFAULT_PARTICIPANTS
    SAMPLING_RATE = DEFAULT_SAMPLING_RATE
    WINDOW_SIZE = DEFAULT_WINDOW_SIZE
    WINDOW_OVERLAP = DEFAULT_WINDOW_OVERLAP
    NUM_CHANNELS = DEFAULT_NUM_CHANNELS
    DATA_TYPE = DEFAULT_DATA_TYPE
    MAX_WINDOWS_PER_PARTICIPANT = DEFAULT_MAX_WINDOWS_PER_PARTICIPANT
    AUTO_INFER_LAYOUT = DEFAULT_AUTO_INFER_LAYOUT
    FREQUENCY_BANDS = DEFAULT_BANDS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def is_probably_html_text_file(path: Path, max_bytes: int = 2048) -> bool:
    try:
        content = path.read_bytes()[:max_bytes]
    except Exception:
        return False
    lower = content.lower()
    # Heuristic: saved web pages typically start with <!DOCTYPE html> or contain <html
    return b"<!doctype html" in lower or b"<html" in lower


def is_probably_brainvision_header(path: Path, max_bytes: int = 256) -> bool:
    try:
        head = path.read_bytes()[:max_bytes]
    except Exception:
        return False
    text = head.decode("utf-8", errors="ignore")
    # Common BrainVision header signature
    return "Brain Vision Data Exchange Header File" in text


def dtype_from_string(dtype_str: str) -> np.dtype:
    normalized = dtype_str.strip().lower()
    mapping = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported DATA_TYPE '{dtype_str}'. Use one of: {sorted(mapping.keys())}")
    return np.dtype(mapping[normalized])


def suggest_layouts(
    file_size_bytes: int, sampling_rate_hz: int, duration_min_range: Tuple[float, float] = (1.0, 180.0)
) -> List[Dict[str, object]]:
    """Suggest plausible (dtype, channels) combinations based on file size.

    This is only a heuristic to help you set NUM_CHANNELS and DATA_TYPE if .vhdr is not usable.
    """
    candidates: List[Dict[str, object]] = []
    dtypes = [("int16", 2), ("int32", 4), ("float32", 4), ("float64", 8)]
    # Include common EEG montages beyond powers of two.
    channels = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 60, 61, 62, 63, 64, 72, 80, 96, 128]

    for dtype_name, dtype_bytes in dtypes:
        for ch in channels:
            if file_size_bytes % (dtype_bytes * ch) != 0:
                continue
            samples = file_size_bytes // (dtype_bytes * ch)
            duration_s = samples / float(sampling_rate_hz)
            duration_min = duration_s / 60.0
            if duration_min_range[0] <= duration_min <= duration_min_range[1]:
                candidates.append(
                    {
                        "dtype": dtype_name,
                        "channels": ch,
                        "samples_per_channel": int(samples),
                        "duration_min": float(duration_min),
                    }
                )

    # Sort by channels (descending) then duration (ascending) as a simple preference.
    candidates.sort(key=lambda x: (-int(x["channels"]), float(x["duration_min"])))
    return candidates


def pick_best_layout(layouts: List[Dict[str, object]], preferred_duration_min: float = 60.0) -> Optional[Dict[str, object]]:
    """Pick a single layout suggestion deterministically.

    Heuristic: prefer more channels (real EEG is usually multi-channel), then duration close to 60 minutes.
    """
    if not layouts:
        return None

    def score(item: Dict[str, object]) -> Tuple[int, float]:
        channels = int(item["channels"])
        duration_min = float(item["duration_min"])
        return (channels, -abs(duration_min - preferred_duration_min))

    return max(layouts, key=score)


def sliding_windows(n_samples: int, window_size: int, overlap: float) -> Iterable[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0)")

    step = max(1, int(round(window_size * (1.0 - overlap))))
    start = 0
    while start + window_size <= n_samples:
        yield start, start + window_size
        start += step


def bandpower_welch(x: np.ndarray, fs: int, band: Tuple[float, float]) -> float:
    """Band power via Welch PSD integral."""
    if x.size == 0:
        return float("nan")
    freqs, psd = welch(x, fs=fs, nperseg=min(256, x.size), noverlap=None)
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return float("nan")
    # NumPy 2.x removed `np.trapz` in favor of `np.trapezoid`.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.trapz(psd[mask], freqs[mask]))


def compute_features_for_window(window: np.ndarray, fs: int, bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Compute basic time + frequency features for a 1D EEG window."""
    x = window.astype(np.float64, copy=False)

    features: Dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "var": float(np.var(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "skew": float(skew(x, bias=False)) if x.size >= 3 else float("nan"),
        "kurtosis": float(kurtosis(x, bias=False)) if x.size >= 4 else float("nan"),
    }

    # Frequency band powers.
    for band_name, band_range in bands.items():
        features[f"bandpower_{band_name}"] = bandpower_welch(x, fs=fs, band=band_range)

    # A couple of simple ratios commonly used in EEG work.
    alpha = features.get("bandpower_alpha", float("nan"))
    beta = features.get("bandpower_beta", float("nan"))
    theta = features.get("bandpower_theta", float("nan"))

    eps = 1e-12
    features["ratio_alpha_beta"] = float(alpha / (beta + eps)) if np.isfinite(alpha) and np.isfinite(beta) else float("nan")
    features["ratio_theta_alpha"] = float(theta / (alpha + eps)) if np.isfinite(theta) and np.isfinite(alpha) else float("nan")

    return features


@dataclass(frozen=True)
class ParticipantFiles:
    participant_id: str
    eeg_path: Path
    vhdr_path: Path
    vmrk_path: Path


def find_participant_files(data_dir: Path, participant_id: str) -> ParticipantFiles:
    # Your folder naming matches `EEG data/Participant-2/` etc.
    # We map P2 -> Participant-2.
    if participant_id.upper().startswith("P") and participant_id[1:].isdigit():
        folder = data_dir / f"Participant-{int(participant_id[1:])}"
        base = participant_id.upper()
    else:
        # Fallback: assume a direct folder exists.
        folder = data_dir / participant_id
        base = participant_id

    return ParticipantFiles(
        participant_id=participant_id,
        eeg_path=folder / f"{base}.eeg",
        vhdr_path=folder / f"{base}.vhdr",
        vmrk_path=folder / f"{base}.vmrk",
    )


def discover_participants(data_dir: Path) -> List[str]:
    """Discover participant IDs by scanning typical folder patterns.

    Supports folders like:
      EEG data/Participant-2/ (maps to P2)
      EEG data/Participant-28/ (maps to P28)
    """

    if not data_dir.exists():
        return []

    participants: List[str] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.lower().startswith("participant-"):
            suffix = name.split("-", 1)[1]
            if suffix.isdigit():
                participants.append(f"P{int(suffix)}")
    return participants


def load_eeg_fallback_binary(eeg_path: Path, dtype: np.dtype, num_channels: int) -> np.ndarray:
    """Fallback binary reader: returns shape (channels, samples)."""
    raw = np.fromfile(eeg_path, dtype=dtype)
    if raw.size == 0:
        raise ValueError(f"Empty EEG file: {eeg_path}")
    if raw.size % num_channels != 0:
        raise ValueError(
            "Cannot reshape EEG data: total values is not divisible by NUM_CHANNELS. "
            f"values={raw.size}, NUM_CHANNELS={num_channels}. "
            "Fix NUM_CHANNELS/DATA_TYPE or re-download valid .vhdr metadata."
        )
    samples = raw.size // num_channels
    data = raw.reshape((samples, num_channels)).T
    return data


def extract_window_features(
    data: np.ndarray,
    fs: int,
    window_size: int,
    overlap: float,
    bands: Dict[str, Tuple[float, float]],
    participant_id: str,
) -> pd.DataFrame:
    """Extract window features from multi-channel data.

    For Phase 1, we aggregate across channels by computing features on the channel-average signal.
    This yields a stable tabular dataset and is computationally feasible for large recordings.
    """

    if data.ndim != 2:
        raise ValueError("Expected data with shape (channels, samples)")

    n_channels, n_samples = data.shape
    rows: List[Dict[str, object]] = []

    # Enumerate all windows, then (optionally) subsample deterministically.
    all_windows = list(sliding_windows(n_samples, window_size, overlap))
    if MAX_WINDOWS_PER_PARTICIPANT is not None and MAX_WINDOWS_PER_PARTICIPANT > 0:
        if len(all_windows) > MAX_WINDOWS_PER_PARTICIPANT:
            # Evenly spaced indices (deterministic) to keep runtime manageable.
            idx = np.linspace(0, len(all_windows) - 1, MAX_WINDOWS_PER_PARTICIPANT).round().astype(int)
            all_windows = [all_windows[i] for i in idx]

    for window_index, (start, end) in enumerate(all_windows):
        window = data[:, start:end]

        # Fast aggregation: compute features on channel-average signal.
        # This is appropriate for Phase 1 exploratory feature engineering.
        avg_signal = window.mean(axis=0)
        averaged = compute_features_for_window(avg_signal, fs, bands)

        row: Dict[str, object] = {
            "participant_id": participant_id,
            "window_index": window_index,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "window_start_sec": float(start / fs),
            "window_end_sec": float(end / fs),
            "n_channels": int(n_channels),
            "fs_hz": int(fs),
        }
        row.update(averaged)
        rows.append(row)

    return pd.DataFrame(rows)


def write_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histogram grid (keep it readable)
    histogram_cols = [
        "bandpower_alpha",
        "bandpower_beta",
        "bandpower_theta",
        "std",
        "rms",
    ]
    histogram_cols = [c for c in histogram_cols if c in df.columns]

    if histogram_cols:
        fig, axes = plt.subplots(1, len(histogram_cols), figsize=(4 * len(histogram_cols), 3), tight_layout=True)
        if len(histogram_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, histogram_cols):
            sns.histplot(df[col].replace([np.inf, -np.inf], np.nan).dropna(), bins=30, ax=ax)
            ax.set_title(col)
        fig.savefig(out_dir / "plot_histograms.png", dpi=150)
        plt.close(fig)

    # Windows per participant bar chart
    fig = plt.figure(figsize=(5, 3), tight_layout=True)
    counts = df.groupby("participant_id")["window_index"].count().sort_values(ascending=False)
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Windows per participant")
    plt.xlabel("participant_id")
    plt.ylabel("n_windows")
    fig.savefig(out_dir / "plot_windows_per_participant.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap (numeric columns only, cap to keep readable)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Drop columns that are purely indexing/time to focus on features.
    numeric_cols = [c for c in numeric_cols if c not in {"window_index", "window_start_sample", "window_end_sample"}]

    # If too many, pick the most variable columns.
    if len(numeric_cols) > 30:
        variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
        numeric_cols = list(variances.head(30).index)

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = plt.figure(figsize=(10, 8), tight_layout=True)
        sns.heatmap(corr, cmap="vlag", center=0, square=True)
        plt.title("Feature correlation (subset)")
        fig.savefig(out_dir / "plot_corr_heatmap.png", dpi=150)
        plt.close(fig)


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    out_dir = workspace_root / "outputs" / "phase1_data_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    participants = PARTICIPANTS
    # If the project config does not explicitly set PARTICIPANTS, auto-discover.
    if participants is None or len(participants) == 0:
        discovered = discover_participants(DATA_DIR)
        if discovered:
            participants = discovered

    dataset_rows: List[pd.DataFrame] = []
    file_audit_rows: List[Dict[str, object]] = []

    for pid in participants:
        files = find_participant_files(DATA_DIR, pid)

        # File audit
        eeg_exists = files.eeg_path.exists()
        vhdr_exists = files.vhdr_path.exists()
        vmrk_exists = files.vmrk_path.exists()

        vhdr_is_html = vhdr_exists and is_probably_html_text_file(files.vhdr_path)
        vmrk_is_html = vmrk_exists and is_probably_html_text_file(files.vmrk_path)
        vhdr_is_brainvision = vhdr_exists and is_probably_brainvision_header(files.vhdr_path)

        audit: Dict[str, object] = {
            "participant_id": pid,
            "eeg_path": str(files.eeg_path),
            "vhdr_path": str(files.vhdr_path),
            "vmrk_path": str(files.vmrk_path),
            "eeg_exists": bool(eeg_exists),
            "vhdr_exists": bool(vhdr_exists),
            "vmrk_exists": bool(vmrk_exists),
            "vhdr_is_html": bool(vhdr_is_html),
            "vmrk_is_html": bool(vmrk_is_html),
            "vhdr_is_brainvision": bool(vhdr_is_brainvision),
        }

        inferred_layout: Optional[Dict[str, object]] = None
        if eeg_exists:
            audit["eeg_size_bytes"] = int(files.eeg_path.stat().st_size)
            suggestions = suggest_layouts(int(audit["eeg_size_bytes"]), SAMPLING_RATE)
            audit["layout_suggestions"] = suggestions[:5]
            inferred_layout = pick_best_layout(suggestions)
            audit["inferred_layout_best"] = inferred_layout

        file_audit_rows.append(audit)

        if not eeg_exists:
            continue

        # Load EEG
        data: Optional[np.ndarray] = None
        loader_used = "fallback_binary"
        assumed_dtype = DATA_TYPE
        assumed_num_channels = NUM_CHANNELS

        # Try MNE BrainVision only if header looks valid.
        if vhdr_is_brainvision:
            try:
                import mne

                raw = mne.io.read_raw_brainvision(str(files.vhdr_path), preload=True, verbose="ERROR")
                data = raw.get_data()  # shape (channels, samples)
                loader_used = "mne_brainvision"
            except Exception:
                data = None

        if data is None:
            if AUTO_INFER_LAYOUT and inferred_layout is not None:
                assumed_dtype = str(inferred_layout["dtype"])
                assumed_num_channels = int(inferred_layout["channels"])

            dtype = dtype_from_string(assumed_dtype)
            data = load_eeg_fallback_binary(files.eeg_path, dtype=dtype, num_channels=assumed_num_channels)

        # Extract features
        df_features = extract_window_features(
            data=data,
            fs=SAMPLING_RATE,
            window_size=WINDOW_SIZE,
            overlap=WINDOW_OVERLAP,
            bands=FREQUENCY_BANDS,
            participant_id=pid,
        )
        df_features["loader_used"] = loader_used
        df_features["assumed_dtype"] = assumed_dtype
        df_features["assumed_num_channels"] = int(assumed_num_channels)
        dataset_rows.append(df_features)

    df_all = pd.concat(dataset_rows, ignore_index=True) if dataset_rows else pd.DataFrame()
    df_audit = pd.DataFrame(file_audit_rows)

    # Save audit and features
    df_audit.to_json(out_dir / "file_audit.json", orient="records", indent=2)
    if not df_all.empty:
        df_all.to_csv(out_dir / "features.csv", index=False)

        # Missingness summary (per column)
        missingness = (
            df_all.isna().mean()
            .reset_index()
            .rename(columns={"index": "column", 0: "missing_fraction"})
            .sort_values("missing_fraction", ascending=False)
        )
        missingness.to_csv(out_dir / "missingness.csv", index=False)

        # Summary stats
        numeric_cols = [c for c in df_all.columns if pd.api.types.is_numeric_dtype(df_all[c])]
        summary_stats = df_all[numeric_cols].describe().T.reset_index().rename(columns={"index": "column"})
        summary_stats.to_csv(out_dir / "summary_stats.csv", index=False)

        # Duplicate checks
        key_cols = ["participant_id", "window_index", "window_start_sample", "window_end_sample"]
        dup_mask = df_all.duplicated(subset=key_cols, keep=False)
        duplicates = df_all.loc[dup_mask, key_cols].sort_values(key_cols)
        duplicates.to_csv(out_dir / "duplicates.csv", index=False)

        # Outlier checks (simple, transparent): per numeric feature, flag |z| > 4
        outlier_rows: List[Dict[str, object]] = []
        for col in numeric_cols:
            series = df_all[col].replace([np.inf, -np.inf], np.nan).dropna()
            if series.size < 10:
                continue
            mean = float(series.mean())
            std = float(series.std(ddof=0))
            if std <= 0:
                continue
            z = (series - mean) / std
            outlier_count = int((np.abs(z) > 4.0).sum())
            outlier_rows.append({"column": col, "outliers_abs_z_gt_4": outlier_count, "n": int(series.size)})

        pd.DataFrame(outlier_rows).sort_values("outliers_abs_z_gt_4", ascending=False).to_csv(
            out_dir / "outliers_summary.csv", index=False
        )

        # Example rows
        df_all.head(10).to_csv(out_dir / "example_rows.csv", index=False)

        # Plots
        write_plots(df_all, out_dir)

    # Console summary for the rubric
    print("=== Phase 1 Data Selection: File Audit ===")
    print(df_audit[["participant_id", "eeg_exists", "vhdr_is_html", "vmrk_is_html", "vhdr_is_brainvision"]])
    print(f"\nWrote outputs to: {out_dir}")
    print("\nIf vhdr_is_html==True, re-download raw BrainVision files (not the web page HTML).")


if __name__ == "__main__":
    main()

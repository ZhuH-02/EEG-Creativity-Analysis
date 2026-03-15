from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _metric_label(metric_col: str) -> str:
    return metric_col.replace("_", " ").upper() if metric_col.lower().endswith("f1") else metric_col.replace("_", " ").title()


def _normalize_label(value: object) -> str:
    text = str(value).strip().replace("_", " ")
    return " ".join(token.capitalize() for token in text.split())


def _coerce_numeric_series(series: pd.Series) -> Tuple[pd.Series, bool]:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric, bool(numeric.notna().all())


def plot_reliability_diagram(
    calibration_df: pd.DataFrame,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration_df["bin_confidence"],
        calibration_df["bin_accuracy"],
        marker="o",
        linewidth=2.0,
        color="#005f73",
        label="Model",
    )
    ax.bar(
        calibration_df["bin_confidence"],
        calibration_df["bin_fraction"],
        width=0.08,
        color="#94d2bd",
        alpha=0.35,
        label="Bin fraction",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy / Fraction")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _style_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confidence_histogram(
    confidences: np.ndarray,
    correct_mask: Optional[np.ndarray] = None,
    title: str = "Confidence Histogram",
    bins: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    conf = np.asarray(confidences, dtype=np.float64)
    if correct_mask is None:
        ax.hist(conf, bins=bins, color="#0a9396", alpha=0.85)
    else:
        correct = conf[np.asarray(correct_mask, dtype=bool)]
        incorrect = conf[~np.asarray(correct_mask, dtype=bool)]
        ax.hist([correct, incorrect], bins=bins, stacked=True, color=["#0a9396", "#ae2012"], label=["Correct", "Incorrect"])
        ax.legend()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_robustness_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    series_col: Optional[str] = None,
    title: str = "Robustness Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plot_df = df.copy()
    if series_col and series_col in plot_df.columns:
        for series_name, group in plot_df.groupby(series_col):
            ax.plot(group[x_col], group[y_col], marker="o", linewidth=2.0, label=str(series_name))
        ax.legend()
    else:
        ax.plot(plot_df[x_col], plot_df[y_col], marker="o", linewidth=2.0, color="#ca6702")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_monitoring_dashboard(
    monitoring_df: pd.DataFrame,
    title: str = "Monitoring Dashboard",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = monitoring_df["batch_index"]

    axes[0, 0].plot(x, monitoring_df["drift_score"], marker="o", color="#bb3e03")
    axes[0, 0].set_title("Drift Score by Batch")
    axes[0, 0].set_ylabel("Score")
    _style_axes(axes[0, 0])

    axes[0, 1].plot(x, monitoring_df["mean_confidence"], marker="o", color="#005f73")
    axes[0, 1].set_title("Mean Confidence")
    axes[0, 1].set_ylabel("Confidence")
    _style_axes(axes[0, 1])

    axes[1, 0].plot(x, monitoring_df["abstention_rate"], marker="o", color="#9b2226")
    axes[1, 0].set_title("Abstention Rate")
    axes[1, 0].set_xlabel("Batch")
    axes[1, 0].set_ylabel("Rate")
    _style_axes(axes[1, 0])

    metric_col = "rolling_macro_f1" if "rolling_macro_f1" in monitoring_df.columns else "rolling_accuracy"
    axes[1, 1].plot(x, monitoring_df[metric_col], marker="o", color="#0a9396")
    axes[1, 1].set_title(metric_col.replace("_", " ").title())
    axes[1, 1].set_xlabel("Batch")
    axes[1, 1].set_ylabel("Metric")
    _style_axes(axes[1, 1])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_adaptation_comparison(
    adaptation_df: pd.DataFrame,
    metric_col: Optional[str] = None,
    title: str = "Adaptation Before vs After",
) -> plt.Figure:
    """Plot before/after adaptation performance for clean and drifted evaluation sets."""
    plot_df = adaptation_df.copy()
    dataset_col = _first_present(plot_df.columns, ["dataset", "split", "condition"])
    stage_col = _first_present(plot_df.columns, ["stage", "phase"])
    if dataset_col is None or stage_col is None:
        raise ValueError("Adaptation comparison requires dataset and stage columns.")

    metric = metric_col or _first_present(plot_df.columns, ["macro_f1", "accuracy", "weighted_f1"])
    if metric is None:
        raise ValueError("Adaptation comparison requires one of: macro_f1, accuracy, weighted_f1.")

    ordered_pairs = [
        ("clean_test", "before", "clean_before"),
        ("clean_test", "after", "clean_after"),
        ("drifted_test", "before", "drifted_before"),
        ("drifted_test", "after", "drifted_after"),
    ]
    labels: List[str] = []
    values: List[float] = []
    for dataset_name, stage_name, label in ordered_pairs:
        mask = (plot_df[dataset_col].astype(str) == dataset_name) & (plot_df[stage_col].astype(str) == stage_name)
        if not mask.any():
            continue
        labels.append(label)
        values.append(float(plot_df.loc[mask, metric].iloc[0]))

    if not labels:
        raise ValueError("Could not find expected clean/drifted before/after rows in adaptation comparison data.")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, width=0.62)
    ax.set_xticks(positions)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels])
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title)
    ax.set_ylim(0.0, max(values) * 1.15 if values else 1.0)
    _style_axes(ax)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_efficiency_comparison(
    efficiency_df: pd.DataFrame,
    title: str = "Efficiency Comparison",
) -> plt.Figure:
    """Plot p50 latency, p90 latency, and throughput across available conditions."""
    plot_df = efficiency_df.copy()
    condition_col = _first_present(plot_df.columns, ["condition", "scenario", "split", "label"])
    if condition_col is None:
        raise ValueError("Efficiency comparison requires a condition column such as scenario or condition.")

    rename_map = {
        "latency_p50_ms": "latency_per_sample_ms_p50",
        "latency_p90_ms": "latency_per_sample_ms_p90",
        "throughput": "throughput_samples_per_sec",
    }
    plot_df = plot_df.rename(columns=rename_map)

    metric_candidates = [
        ("latency_per_sample_ms_p50", "Latency p50 (ms)"),
        ("latency_per_sample_ms_p90", "Latency p90 (ms)"),
        ("throughput_samples_per_sec", "Throughput (samples/sec)"),
    ]
    available_metrics = [(column, label) for column, label in metric_candidates if column in plot_df.columns]
    if not available_metrics:
        raise ValueError("Efficiency comparison requires latency or throughput columns.")

    conditions = plot_df[condition_col].astype(str).tolist()
    positions = np.arange(len(conditions), dtype=float)
    width = 0.22 if len(available_metrics) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(max(8.0, len(conditions) * 1.4), 5.0))
    for idx, (metric_col, legend_label) in enumerate(available_metrics):
        offsets = positions + (idx - (len(available_metrics) - 1) / 2.0) * width
        values = pd.to_numeric(plot_df[metric_col], errors="coerce").to_numpy(dtype=np.float64)
        bars = ax.bar(offsets, values, width=width, label=legend_label)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}" if value < 100 else f"{value:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels([_normalize_label(value) for value in conditions], rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    _style_axes(ax)
    if len(available_metrics) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_stress_test_summary(
    stress_df: pd.DataFrame,
    title: str = "Stress Test Summary",
) -> plt.Figure:
    """Plot stress test results as lines by severity when numeric, otherwise grouped bars."""
    plot_df = stress_df.copy()
    scenario_col = _first_present(plot_df.columns, ["scenario", "test_name", "perturbation", "corruption"])
    if scenario_col is None:
        raise ValueError("Stress test summary requires a scenario-like column.")

    metric_col = _first_present(plot_df.columns, ["macro_f1", "accuracy", "weighted_f1"])
    if metric_col is None:
        raise ValueError("Stress test summary requires one of: macro_f1, accuracy, weighted_f1.")

    severity_col = _first_present(plot_df.columns, ["severity", "strength", "epsilon", "ratio"])
    if severity_col is None:
        summary_df = plot_df.groupby(scenario_col, as_index=False)[metric_col].mean()
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        positions = np.arange(len(summary_df))
        bars = ax.bar(positions, summary_df[metric_col].to_numpy(dtype=np.float64))
        ax.set_xticks(positions)
        ax.set_xticklabels([_normalize_label(value) for value in summary_df[scenario_col]], rotation=20, ha="right")
        ax.set_ylabel(_metric_label(metric_col))
        ax.set_title(title)
        _style_axes(ax)
        for bar, value in zip(bars, summary_df[metric_col].tolist()):
            ax.text(bar.get_x() + bar.get_width() / 2.0, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        return fig

    severity_numeric, is_numeric = _coerce_numeric_series(plot_df[severity_col])
    plot_df = plot_df.copy()
    plot_df["_severity_numeric"] = severity_numeric

    if is_numeric:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for scenario_name, group in plot_df.groupby(scenario_col):
            clean_group = group.sort_values("_severity_numeric")
            ax.plot(
                clean_group["_severity_numeric"].to_numpy(dtype=np.float64),
                pd.to_numeric(clean_group[metric_col], errors="coerce").to_numpy(dtype=np.float64),
                marker="o",
                linewidth=2.0,
                label=_normalize_label(scenario_name),
            )
        ax.set_xlabel(_normalize_label(severity_col))
        ax.set_ylabel(_metric_label(metric_col))
        ax.set_title(title)
        _style_axes(ax)
        ax.legend()
        fig.tight_layout()
        return fig

    summary_df = plot_df.groupby(scenario_col, as_index=False)[metric_col].mean().sort_values(metric_col, ascending=False)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(summary_df))
    bars = ax.bar(positions, summary_df[metric_col].to_numpy(dtype=np.float64))
    ax.set_xticks(positions)
    ax.set_xticklabels([_normalize_label(value) for value in summary_df[scenario_col]], rotation=20, ha="right")
    ax.set_ylabel(_metric_label(metric_col))
    ax.set_title(title)
    _style_axes(ax)
    for bar, value in zip(bars, summary_df[metric_col].tolist()):
        ax.text(bar.get_x() + bar.get_width() / 2.0, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def format_summary_table(df: pd.DataFrame, float_precision: int = 4) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.{float_precision}f}")
    return formatted


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

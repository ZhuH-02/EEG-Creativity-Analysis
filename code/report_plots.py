from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


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


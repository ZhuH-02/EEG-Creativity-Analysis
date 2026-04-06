from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import load_json_file, make_output_dir, save_csv, save_json
from report_plots import (
    plot_active_learning_workflow,
    plot_continual_learning_workflow,
    plot_system_architecture,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 final reporting: aggregate Milestone 4 outputs into submission-ready summary artifacts."
    )
    parser.add_argument("--training_run_dir", type=str, default="", help="Training run directory with metrics.json and efficiency.json.")
    parser.add_argument("--phase2_dir", type=str, default="", help="Phase 2 continual learning output directory.")
    parser.add_argument("--phase3_dir", type=str, default="", help="Phase 3 HITL output directory.")
    parser.add_argument("--phase4_dir", type=str, default="", help="Phase 4 active learning output directory.")
    parser.add_argument("--phase5_dir", type=str, default="", help="Phase 5 end-to-end pipeline output directory.")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Base directory used to auto-discover training and Milestone 4 artifacts.")
    parser.add_argument("--output_dir", type=str, default="runs/milestone4", help="Base directory for final report artifacts.")
    parser.add_argument("--run_name", type=str, default="phase6_final_report")
    return parser.parse_args()


def _latest_dir_with_file(base_dir: Path, filename: str) -> Path:
    candidates = [path.parent for path in base_dir.rglob(filename) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"Could not find any '{filename}' under {base_dir}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_training_run(base_dir: Path) -> Path:
    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "metrics.json").exists() and (path / "efficiency.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find a training run under {base_dir}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _coerce_dir(raw: str, fallback: Path, expected_file: str) -> Path:
    if raw:
        candidate = Path(raw)
        if not (candidate / expected_file).exists():
            raise FileNotFoundError(f"Expected '{expected_file}' in {candidate}, but it was not found.")
        return candidate
    return _latest_dir_with_file(fallback, expected_file)


def _safe_get(payload: Dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _best_strategy(active_summary: Dict[str, Any]) -> Dict[str, Any]:
    strategies = active_summary.get("strategies", [])
    if not strategies:
        return {}
    return max(
        strategies,
        key=lambda row: (
            float(row.get("drifted_macro_f1_final", float("-inf"))),
            float(row.get("clean_macro_f1_final", float("-inf"))),
        ),
    )


def _build_summary_rows(
    training_dir: Path,
    training_metrics: Dict[str, Any],
    training_efficiency: Dict[str, Any],
    phase2_dir: Path,
    phase2_summary: Dict[str, Any],
    phase3_dir: Path,
    phase3_summary: Dict[str, Any],
    phase4_dir: Path,
    phase4_summary: Dict[str, Any],
    phase5_dir: Path,
    phase5_summary: Dict[str, Any],
) -> pd.DataFrame:
    baseline_clean_f1 = float(_safe_get(training_metrics, "test", "f1_macro", default=np.nan))
    best_phase4 = _best_strategy(phase4_summary)

    rows: List[Dict[str, Any]] = [
        {
            "stage": "baseline_trainrun",
            "source_dir": str(training_dir.resolve()),
            "clean_accuracy": float(_safe_get(training_metrics, "test", "accuracy", default=np.nan)),
            "clean_macro_f1": baseline_clean_f1,
            "clean_weighted_f1": float(_safe_get(training_metrics, "test", "f1_weighted", default=np.nan)),
            "drifted_accuracy": np.nan,
            "drifted_macro_f1": np.nan,
            "macro_f1_gain_vs_baseline": 0.0,
            "queries_or_reviews": 0,
            "updates_or_cycles": 0,
            "estimated_human_time_sec": 0.0,
            "throughput_samples_per_sec": float(_safe_get(training_efficiency, "inference", "throughput_samples_per_sec", default=np.nan)),
            "peak_ram_mb": float(_safe_get(training_efficiency, "inference", "peak_ram_mb", default=np.nan)),
            "notes": "Milestone 2 baseline test performance",
        },
        {
            "stage": "phase2_continual_learning",
            "source_dir": str(phase2_dir.resolve()),
            "clean_accuracy": np.nan,
            "clean_macro_f1": np.nan,
            "clean_weighted_f1": float(_safe_get(phase2_summary, "final_stream_metrics", "weighted_f1", default=np.nan)),
            "drifted_accuracy": float(_safe_get(phase2_summary, "final_stream_metrics", "accuracy", default=np.nan)),
            "drifted_macro_f1": float(_safe_get(phase2_summary, "final_stream_metrics", "macro_f1", default=np.nan)),
            "macro_f1_gain_vs_baseline": float(_safe_get(phase2_summary, "final_stream_metrics", "macro_f1", default=np.nan)) - baseline_clean_f1,
            "queries_or_reviews": 0,
            "updates_or_cycles": int(_safe_get(phase2_summary, "updates_completed", default=0)),
            "estimated_human_time_sec": 0.0,
            "throughput_samples_per_sec": float(_safe_get(phase2_summary, "efficiency", "final_inference", "throughput_samples_per_sec", default=np.nan)),
            "peak_ram_mb": float(_safe_get(phase2_summary, "efficiency", "final_inference", "peak_ram_mb", default=np.nan)),
            "notes": f"Replay buffer final size={int(_safe_get(phase2_summary, 'buffer', 'final_size', default=0))}",
        },
        {
            "stage": "phase3_hitl",
            "source_dir": str(phase3_dir.resolve()),
            "clean_accuracy": np.nan,
            "clean_macro_f1": np.nan,
            "clean_weighted_f1": np.nan,
            "drifted_accuracy": np.nan,
            "drifted_macro_f1": np.nan,
            "macro_f1_gain_vs_baseline": np.nan,
            "queries_or_reviews": int(_safe_get(phase3_summary, "samples_reviewed", default=0)),
            "updates_or_cycles": int(_safe_get(phase3_summary, "intervention_batches", default=0)),
            "estimated_human_time_sec": np.nan,
            "throughput_samples_per_sec": np.nan,
            "peak_ram_mb": np.nan,
            "notes": (
                "coverage="
                f"{float(_safe_get(phase3_summary, 'review_coverage', default=np.nan)):.4f}, "
                f"correction_rate={float(_safe_get(phase3_summary, 'human_correction_rate', default=np.nan)):.4f}"
            ),
        },
        {
            "stage": "phase4_active_learning_best",
            "source_dir": str(phase4_dir.resolve()),
            "clean_accuracy": np.nan,
            "clean_macro_f1": float(best_phase4.get("clean_macro_f1_final", np.nan)),
            "clean_weighted_f1": np.nan,
            "drifted_accuracy": np.nan,
            "drifted_macro_f1": float(best_phase4.get("drifted_macro_f1_final", np.nan)),
            "macro_f1_gain_vs_baseline": float(best_phase4.get("clean_macro_f1_final", np.nan)) - baseline_clean_f1,
            "queries_or_reviews": int(best_phase4.get("total_labels_reviewed", 0)),
            "updates_or_cycles": int(best_phase4.get("cycles_completed", 0)),
            "estimated_human_time_sec": np.nan,
            "throughput_samples_per_sec": np.nan,
            "peak_ram_mb": np.nan,
            "notes": f"Best strategy={best_phase4.get('strategy', 'n/a')}",
        },
        {
            "stage": "phase5_end_to_end_system",
            "source_dir": str(phase5_dir.resolve()),
            "clean_accuracy": float(_safe_get(phase5_summary, "final_clean", "accuracy", default=np.nan)),
            "clean_macro_f1": float(_safe_get(phase5_summary, "final_clean", "macro_f1", default=np.nan)),
            "clean_weighted_f1": np.nan,
            "drifted_accuracy": float(_safe_get(phase5_summary, "final_drifted", "accuracy", default=np.nan)),
            "drifted_macro_f1": float(_safe_get(phase5_summary, "final_drifted", "macro_f1", default=np.nan)),
            "macro_f1_gain_vs_baseline": float(_safe_get(phase5_summary, "final_clean", "macro_f1", default=np.nan)) - baseline_clean_f1,
            "queries_or_reviews": int(_safe_get(phase5_summary, "system_counts", "total_queries", default=0)),
            "updates_or_cycles": int(_safe_get(phase5_summary, "system_counts", "updates_completed", default=0)),
            "estimated_human_time_sec": float(_safe_get(phase5_summary, "resources", "estimated_human_time_sec", default=np.nan)),
            "throughput_samples_per_sec": float(_safe_get(phase5_summary, "resources", "final_inference", "throughput_samples_per_sec", default=np.nan)),
            "peak_ram_mb": float(_safe_get(phase5_summary, "resources", "final_inference", "peak_ram_mb", default=np.nan)),
            "notes": f"final_model_version={int(_safe_get(phase5_summary, 'system_counts', 'final_model_version', default=0))}",
        },
    ]
    return pd.DataFrame(rows)


def _plot_system_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    clean_rows = summary_df.dropna(subset=["clean_macro_f1"])
    axes[0, 0].bar(clean_rows["stage"], clean_rows["clean_macro_f1"], color="#0a9396")
    axes[0, 0].set_title("Clean Macro-F1")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(alpha=0.25)

    drift_rows = summary_df.dropna(subset=["drifted_macro_f1"])
    axes[0, 1].bar(drift_rows["stage"], drift_rows["drifted_macro_f1"], color="#ca6702")
    axes[0, 1].set_title("Drifted / Stream Macro-F1")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(alpha=0.25)

    interaction_rows = summary_df.fillna({"queries_or_reviews": 0, "updates_or_cycles": 0})
    x = np.arange(len(interaction_rows))
    width = 0.35
    axes[1, 0].bar(x - width / 2.0, interaction_rows["queries_or_reviews"], width=width, color="#005f73", label="Queries / Reviews")
    axes[1, 0].bar(x + width / 2.0, interaction_rows["updates_or_cycles"], width=width, color="#94d2bd", label="Updates / Cycles")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(interaction_rows["stage"], rotation=20, ha="right")
    axes[1, 0].set_title("Operational Load")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    resource_rows = summary_df.dropna(subset=["throughput_samples_per_sec"])
    axes[1, 1].bar(resource_rows["stage"], resource_rows["throughput_samples_per_sec"], color="#6c584c")
    axes[1, 1].set_title("Inference Throughput")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_submission_notes(
    output_path: Path,
    training_dir: Path,
    phase2_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    phase5_dir: Path,
    best_strategy: Dict[str, Any],
    phase5_summary: Dict[str, Any],
) -> None:
    text = "\n".join(
        [
            "# Milestone 4 Submission Notes",
            "",
            "## Source Artifacts",
            f"- Training baseline: `{training_dir}`",
            f"- Phase 2 continual learning: `{phase2_dir}`",
            f"- Phase 3 HITL: `{phase3_dir}`",
            f"- Phase 4 active learning: `{phase4_dir}`",
            f"- Phase 5 end-to-end system: `{phase5_dir}`",
            "",
            "## Key Results",
            f"- Best active learning strategy: `{best_strategy.get('strategy', 'n/a')}`",
            f"- Phase 5 final clean macro-F1: `{float(_safe_get(phase5_summary, 'final_clean', 'macro_f1', default=np.nan)):.4f}`",
            f"- Phase 5 final drifted macro-F1: `{float(_safe_get(phase5_summary, 'final_drifted', 'macro_f1', default=np.nan)):.4f}`",
            f"- Phase 5 total queries: `{int(_safe_get(phase5_summary, 'system_counts', 'total_queries', default=0))}`",
            f"- Phase 5 updates completed: `{int(_safe_get(phase5_summary, 'system_counts', 'updates_completed', default=0))}`",
            "",
            "## Suggested Demo Order",
            "1. Show the baseline training run metrics and efficiency.",
            "2. Show Phase 2 replay-based continual learning outputs.",
            "3. Show Phase 3 HITL candidate and feedback logs.",
            "4. Show Phase 4 active learning curve and strategy summary.",
            "5. Show Phase 5 end-to-end timeline and rollout history.",
            "",
            "## Recommended Files to Mention in the Report",
            "- `project_summary_table.csv`",
            "- `final_system_summary.json`",
            "- `system_comparison.png`",
            "- `continual_learning_workflow.png`",
            "- `active_learning_workflow.png`",
            "- `system_architecture.png`",
            "- `submission_notes.md`",
        ]
    )
    output_path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    milestone4_root = runs_dir / "milestone4"

    training_dir = Path(args.training_run_dir) if args.training_run_dir else _latest_training_run(runs_dir)
    phase2_dir = _coerce_dir(args.phase2_dir, milestone4_root, "continual_summary.json")
    phase3_dir = _coerce_dir(args.phase3_dir, milestone4_root, "review_summary.json")
    phase4_dir = _coerce_dir(args.phase4_dir, milestone4_root, "active_learning_summary.json")
    phase5_dir = _coerce_dir(args.phase5_dir, milestone4_root, "end_to_end_summary.json")

    output_dir = make_output_dir(args.output_dir, args.run_name)

    training_metrics = load_json_file(training_dir / "metrics.json")
    training_efficiency = load_json_file(training_dir / "efficiency.json")
    phase2_summary = load_json_file(phase2_dir / "continual_summary.json")
    phase3_summary = load_json_file(phase3_dir / "review_summary.json")
    phase4_summary = load_json_file(phase4_dir / "active_learning_summary.json")
    phase5_summary = load_json_file(phase5_dir / "end_to_end_summary.json")

    summary_df = _build_summary_rows(
        training_dir=training_dir,
        training_metrics=training_metrics,
        training_efficiency=training_efficiency,
        phase2_dir=phase2_dir,
        phase2_summary=phase2_summary,
        phase3_dir=phase3_dir,
        phase3_summary=phase3_summary,
        phase4_dir=phase4_dir,
        phase4_summary=phase4_summary,
        phase5_dir=phase5_dir,
        phase5_summary=phase5_summary,
    )
    best_strategy = _best_strategy(phase4_summary)

    artifact_manifest = {
        "training_run_dir": str(training_dir.resolve()),
        "phase2_dir": str(phase2_dir.resolve()),
        "phase3_dir": str(phase3_dir.resolve()),
        "phase4_dir": str(phase4_dir.resolve()),
        "phase5_dir": str(phase5_dir.resolve()),
    }
    final_summary = {
        "output_dir": str(output_dir.resolve()),
        "artifact_manifest": artifact_manifest,
        "best_active_learning_strategy": best_strategy,
        "baseline_test": {
            "accuracy": float(_safe_get(training_metrics, "test", "accuracy", default=np.nan)),
            "macro_f1": float(_safe_get(training_metrics, "test", "f1_macro", default=np.nan)),
        },
        "phase5_end_to_end": phase5_summary,
        "phase3_hitl": phase3_summary,
        "phase2_continual_learning": phase2_summary,
        "phase4_active_learning": phase4_summary,
    }

    save_csv(output_dir / "project_summary_table.csv", summary_df)
    save_json(output_dir / "artifact_manifest.json", artifact_manifest)
    save_json(output_dir / "final_system_summary.json", final_summary)
    _plot_system_comparison(summary_df, output_dir / "system_comparison.png")
    save_figure(plot_continual_learning_workflow(), output_dir / "continual_learning_workflow.png")
    save_figure(plot_active_learning_workflow(), output_dir / "active_learning_workflow.png")
    save_figure(plot_system_architecture(), output_dir / "system_architecture.png")
    _write_submission_notes(
        output_path=output_dir / "submission_notes.md",
        training_dir=training_dir,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
        phase4_dir=phase4_dir,
        phase5_dir=phase5_dir,
        best_strategy=best_strategy,
        phase5_summary=phase5_summary,
    )

    print(
        {
            "output_dir": str(output_dir),
            "best_active_strategy": best_strategy.get("strategy", "n/a"),
            "summary_rows": int(len(summary_df)),
        }
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from analysis_utils import (
    compute_common_metrics,
    confidence_threshold_predictions,
    evaluate_model,
    load_dataset_for_run,
    load_trained_artifacts,
    make_output_dir,
    save_csv,
    save_json,
    select_key_features,
    set_seed,
)
from report_plots import plot_monitoring_dashboard, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate offline production monitoring on batched EEG inference.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/final_eval")
    parser.add_argument("--run_name", type=str, default="monitoring_sim")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reference_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stream_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--stream_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--rolling_window_batches", type=int, default=3)
    parser.add_argument("--psi_warning", type=float, default=0.2)
    parser.add_argument("--psi_critical", type=float, default=0.3)
    parser.add_argument("--confidence_drop", type=float, default=0.08)
    parser.add_argument("--abstention_spike", type=float, default=0.12)
    parser.add_argument("--metric_drop", type=float, default=0.1)
    parser.add_argument("--top_k_features", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(np.sum(ref_hist), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(np.sum(cur_hist), 1), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def batched_indices(n_samples: int, batch_size: int) -> List[np.ndarray]:
    indices = np.arange(n_samples, dtype=int)
    return [indices[start : start + batch_size] for start in range(0, n_samples, batch_size)]


def aggregate_batch_drift(reference_df: pd.DataFrame, batch_df: pd.DataFrame, feature_names: Sequence[str], bins: int) -> Tuple[float, List[Dict[str, Any]]]:
    feature_rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for feature in feature_names:
        reference_values = reference_df[feature].to_numpy(dtype=np.float64)
        batch_values = batch_df[feature].to_numpy(dtype=np.float64)
        psi = population_stability_index(reference_values, batch_values, bins=bins)
        wd = float(wasserstein_distance(reference_values, batch_values))
        mean_shift = float(abs(np.mean(batch_values) - np.mean(reference_values)))
        ref_std = max(float(np.std(reference_values)), 1e-8)
        std_ratio = float(np.std(batch_values) / ref_std)
        score = psi + wd / max(ref_std, 1e-8)
        scores.append(score)
        feature_rows.append(
            {
                "feature": feature,
                "psi": psi,
                "wasserstein": wd,
                "mean_shift": mean_shift,
                "std_ratio": std_ratio,
                "drift_score": score,
            }
        )
    return float(np.mean(scores)) if scores else 0.0, feature_rows


def rolling_slice(history: List[np.ndarray], window: int) -> np.ndarray:
    tail = history[-int(window) :] if int(window) > 0 else history
    return np.concatenate(tail, axis=0) if tail else np.zeros((0,), dtype=np.int64)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir, args.run_name)

    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_ref, y_ref, _ = dataset.subset(args.reference_split)
    X_stream, y_stream, groups_stream = dataset.subset(args.stream_split)

    reference_df = pd.DataFrame(X_ref, columns=bundle.feature_names)
    key_features = select_key_features(X_ref, bundle.feature_names, top_k=args.top_k_features)
    stream_batches = batched_indices(len(X_stream), args.stream_batch_size)

    reference_result = evaluate_model(
        model=bundle.model,
        X=X_ref,
        y=y_ref,
        device=bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )
    reference_predictions, reference_accepted = confidence_threshold_predictions(
        reference_result["probabilities"],
        threshold=args.confidence_threshold,
    )
    baseline_confidence = float(np.mean(reference_result["confidences"])) if len(reference_result["confidences"]) else 0.0
    baseline_abstention = float(1.0 - np.mean(reference_accepted)) if len(reference_accepted) else 0.0
    baseline_macro_f1 = float(reference_result["macro_f1"])
    baseline_accuracy = float(reference_result["accuracy"])

    monitoring_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []
    y_history: List[np.ndarray] = []
    pred_history: List[np.ndarray] = []
    prob_history: List[np.ndarray] = []

    for batch_index, idx in enumerate(stream_batches, start=1):
        X_batch = X_stream[idx]
        y_batch = y_stream[idx]
        groups_batch = groups_stream[idx]
        batch_df = pd.DataFrame(X_batch, columns=bundle.feature_names)
        batch_result = evaluate_model(
            model=bundle.model,
            X=X_batch,
            y=y_batch,
            device=bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=bundle.scaler,
            labels=bundle.labels,
            class_names=bundle.class_names,
        )
        predictions, accepted_mask = confidence_threshold_predictions(
            batch_result["probabilities"],
            threshold=args.confidence_threshold,
        )
        y_history.append(np.asarray(y_batch, dtype=np.int64))
        pred_history.append(np.asarray(predictions, dtype=np.int64))
        prob_history.append(np.asarray(batch_result["probabilities"], dtype=np.float64))
        rolling_y = rolling_slice(y_history, args.rolling_window_batches)
        rolling_pred = rolling_slice(pred_history, args.rolling_window_batches)
        rolling_probs = rolling_slice(prob_history, args.rolling_window_batches)
        rolling_metrics = compute_common_metrics(
            y_true=rolling_y,
            y_pred=rolling_pred,
            probabilities=rolling_probs,
            labels=bundle.labels,
            class_names=bundle.class_names,
        )

        drift_score, feature_rows = aggregate_batch_drift(reference_df, batch_df, key_features, bins=args.bins)
        for feature_row in feature_rows:
            drift_rows.append(
                {
                    "batch_index": batch_index,
                    "participant_examples": ",".join(sorted(set(groups_batch.astype(str).tolist()))[:3]),
                    **feature_row,
                }
            )

        batch_row: Dict[str, Any] = {
            "batch_index": batch_index,
            "n_samples": int(len(idx)),
            "participants": ",".join(sorted(set(groups_batch.astype(str).tolist()))),
            "null_rate": float(batch_df.isna().mean().mean()),
            "mean_confidence": float(np.mean(batch_result["confidences"])) if len(batch_result["confidences"]) else 0.0,
            "abstention_rate": float(1.0 - np.mean(accepted_mask)) if len(accepted_mask) else 0.0,
            "rolling_accuracy": float(rolling_metrics["accuracy"]),
            "rolling_macro_f1": float(rolling_metrics["macro_f1"]),
            "drift_score": drift_score,
        }
        for label in bundle.labels:
            batch_row[f"pred_dist_{label}"] = float(np.mean(predictions == int(label))) if len(predictions) else 0.0
        monitoring_rows.append(batch_row)

        max_psi = max((row["psi"] for row in feature_rows), default=0.0)
        if max_psi > args.psi_warning:
            alerts.append(
                {
                    "batch_index": batch_index,
                    "level": "critical" if max_psi > args.psi_critical else "warning",
                    "type": "psi",
                    "message": f"PSI reached {max_psi:.3f}",
                }
            )
        if batch_row["mean_confidence"] < baseline_confidence - args.confidence_drop:
            alerts.append(
                {
                    "batch_index": batch_index,
                    "level": "warning",
                    "type": "confidence_drop",
                    "message": f"Mean confidence fell from {baseline_confidence:.3f} to {batch_row['mean_confidence']:.3f}",
                }
            )
        if batch_row["abstention_rate"] > baseline_abstention + args.abstention_spike:
            alerts.append(
                {
                    "batch_index": batch_index,
                    "level": "warning",
                    "type": "abstention_spike",
                    "message": f"Abstention rate increased to {batch_row['abstention_rate']:.3f}",
                }
            )
        if batch_row["rolling_macro_f1"] < baseline_macro_f1 - args.metric_drop:
            alerts.append(
                {
                    "batch_index": batch_index,
                    "level": "critical",
                    "type": "rolling_metric_degradation",
                    "message": f"Rolling macro-F1 dropped from {baseline_macro_f1:.3f} to {batch_row['rolling_macro_f1']:.3f}",
                }
            )

    monitoring_df = pd.DataFrame(monitoring_rows)
    drift_df = pd.DataFrame(drift_rows)
    dashboard = plot_monitoring_dashboard(monitoring_df, title="Offline Monitoring Simulation")

    save_csv(output_dir / "monitoring_log.csv", monitoring_df)
    save_csv(output_dir / "drift_metrics.csv", drift_df)
    save_json(
        output_dir / "alerts.json",
        {
            "reference_split": args.reference_split,
            "stream_split": args.stream_split,
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_f1": baseline_macro_f1,
            "baseline_mean_confidence": baseline_confidence,
            "alerts": alerts,
        },
    )
    save_figure(dashboard, output_dir / "monitoring_dashboard.png")
    (output_dir / "monitoring_summary.txt").write_text(
        textwrap.dedent(
            f"""
            Monitoring simulation complete.
            Output directory: {output_dir}
            Batches processed: {len(monitoring_df)}
            Alerts raised: {len(alerts)}
            Mean drift score: {monitoring_df['drift_score'].mean():.4f}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Monitoring simulation complete. Output directory: {output_dir}. "
        f"Batches={len(monitoring_df)} Alerts={len(alerts)}"
    )


if __name__ == "__main__":
    main()

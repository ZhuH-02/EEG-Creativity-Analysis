from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptation_eval import apply_class_prior_shift, simulate_feature_drift
from analysis_utils import (
    confidence_threshold_predictions,
    evaluate_model,
    extract_softmax_confidences,
    load_dataset_for_run,
    load_trained_artifacts,
    make_output_dir,
    save_csv,
    save_json,
    select_key_features,
    set_seed,
    summarize_feature_reference,
)
from failure_checks import (
    compute_anomaly_scores,
    low_confidence_flags,
    suspicious_high_confidence_on_anomalies,
)
from monitoring_sim import aggregate_batch_drift, batched_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4 Phase 3: HITL trigger logic and human feedback simulation."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/milestone4")
    parser.add_argument("--run_name", type=str, default="phase3_hitl")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reference_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stream_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--stream_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k_features", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--drift_start_batch", type=int, default=4)
    parser.add_argument("--drift_noise", type=float, default=0.05)
    parser.add_argument("--drift_scale", type=float, default=1.2)
    parser.add_argument("--drift_mean_shift", type=float, default=0.25)
    parser.add_argument("--class_prior_strength", type=float, default=0.0)
    parser.add_argument("--drift_trigger_threshold", type=float, default=1.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.55)
    parser.add_argument("--high_confidence_threshold", type=float, default=0.85)
    parser.add_argument("--anomaly_threshold", type=float, default=4.0)
    parser.add_argument("--low_conf_fraction_trigger", type=float, default=0.35)
    parser.add_argument("--query_budget_per_batch", type=int, default=24)
    parser.add_argument("--annotator_accuracy", type=float, default=1.0)
    return parser.parse_args()


def select_hitl_candidates(
    batch_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    anomaly_scores: np.ndarray,
    drift_score: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    low_conf_mask = low_confidence_flags(confidences, threshold=args.confidence_threshold)
    suspicious_mask = suspicious_high_confidence_on_anomalies(
        anomaly_scores=anomaly_scores,
        confidences=confidences,
        anomaly_threshold=args.anomaly_threshold,
        confidence_threshold=args.high_confidence_threshold,
    )
    error_mask = np.asarray(y_true, dtype=np.int64) != np.asarray(y_pred, dtype=np.int64)
    overconfident_error_mask = error_mask & (np.asarray(confidences, dtype=np.float64) >= float(args.high_confidence_threshold))

    candidate_rows: List[Dict[str, Any]] = []
    for row_idx in range(len(batch_df)):
        reasons: List[str] = []
        priority = 0.0
        if bool(low_conf_mask[row_idx]):
            reasons.append("low_confidence")
            priority += float(1.0 - confidences[row_idx]) * 2.0
        if bool(suspicious_mask[row_idx]):
            reasons.append("anomalous_high_confidence")
            priority += float(anomaly_scores[row_idx]) * 1.5
        if bool(overconfident_error_mask[row_idx]):
            reasons.append("overconfident_error")
            priority += 5.0
        if drift_score >= float(args.drift_trigger_threshold):
            reasons.append("drift_batch")
            priority += float(drift_score) * 0.5

        if not reasons:
            continue

        candidate_rows.append(
            {
                "row_index_in_batch": int(row_idx),
                "y_true": int(y_true[row_idx]),
                "y_pred": int(y_pred[row_idx]),
                "confidence": float(confidences[row_idx]),
                "anomaly_score": float(anomaly_scores[row_idx]),
                "model_error": int(error_mask[row_idx]),
                "priority_score": float(priority),
                "selection_reasons": ",".join(reasons),
            }
        )

    if not candidate_rows and drift_score >= float(args.drift_trigger_threshold):
        fallback_order = np.argsort(confidences)[: int(max(1, args.query_budget_per_batch))]
        for row_idx in fallback_order.tolist():
            candidate_rows.append(
                {
                    "row_index_in_batch": int(row_idx),
                    "y_true": int(y_true[row_idx]),
                    "y_pred": int(y_pred[row_idx]),
                    "confidence": float(confidences[row_idx]),
                    "anomaly_score": float(anomaly_scores[row_idx]),
                    "model_error": int(error_mask[row_idx]),
                    "priority_score": float(1.0 - confidences[row_idx]),
                    "selection_reasons": "drift_batch_fallback",
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        return candidate_df

    candidate_df = candidate_df.sort_values(
        ["priority_score", "model_error", "confidence"],
        ascending=[False, False, True],
    ).head(int(args.query_budget_per_batch))
    return candidate_df.reset_index(drop=True)


def simulate_human_feedback(
    candidate_df: pd.DataFrame,
    annotator_accuracy: float,
    labels: List[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()

    feedback_rows: List[Dict[str, Any]] = []
    label_choices = [int(v) for v in labels]
    for row in candidate_df.to_dict(orient="records"):
        y_true = int(row["y_true"])
        y_pred = int(row["y_pred"])
        human_correct = bool(rng.random() <= float(annotator_accuracy))
        if human_correct:
            reviewed_label = y_true
        else:
            alternatives = [label for label in label_choices if label != y_true]
            reviewed_label = int(rng.choice(alternatives)) if alternatives else y_true

        feedback_rows.append(
            {
                **row,
                "reviewed_label": int(reviewed_label),
                "human_correct": int(human_correct),
                "model_corrected": int(reviewed_label != y_pred),
                "human_agrees_with_model": int(reviewed_label == y_pred),
            }
        )

    return pd.DataFrame(feedback_rows)


def plot_hitl_timeline(intervention_df: pd.DataFrame, output_path: Path) -> None:
    if intervention_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = intervention_df["batch_index"].to_numpy(dtype=np.int64)

    axes[0].plot(x, intervention_df["drift_score"], marker="o", color="#bb3e03", label="Drift score")
    axes[0].axhline(float(intervention_df["drift_threshold"].iloc[0]), linestyle="--", color="#9b2226", alpha=0.6, label="Drift threshold")
    axes[0].set_ylabel("Drift score")
    axes[0].set_title("HITL Trigger Timeline")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, intervention_df["selected_for_review"], color="#0a9396", label="Samples sent to review")
    axes[1].plot(x, intervention_df["low_conf_fraction"], marker="o", color="#005f73", label="Low-conf fraction")
    axes[1].axhline(float(intervention_df["low_conf_fraction_trigger"].iloc[0]), linestyle="--", color="#94d2bd", alpha=0.6, label="Low-conf trigger")
    axes[1].set_xlabel("Batch index")
    axes[1].set_ylabel("Review count / fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = make_output_dir(args.output_dir, args.run_name)

    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_reference, y_reference, _ = dataset.subset(args.reference_split)
    X_stream, y_stream, groups_stream = dataset.subset(args.stream_split)

    reference_stats = summarize_feature_reference(X_reference, bundle.feature_names)
    reference_df = pd.DataFrame(X_reference, columns=bundle.feature_names)
    key_features = select_key_features(X_reference, bundle.feature_names, top_k=args.top_k_features)
    stream_batches = batched_indices(len(X_stream), args.stream_batch_size)

    intervention_rows: List[Dict[str, Any]] = []
    candidate_frames: List[pd.DataFrame] = []
    feedback_frames: List[pd.DataFrame] = []

    total_reviewed = 0
    total_corrected = 0

    for batch_index, idx in enumerate(stream_batches, start=1):
        X_batch = np.asarray(X_stream[idx], dtype=np.float64)
        y_batch = np.asarray(y_stream[idx], dtype=np.int64)
        groups_batch = np.asarray(groups_stream[idx], dtype=object)

        drift_applied = batch_index >= int(args.drift_start_batch)
        if drift_applied:
            X_batch = simulate_feature_drift(
                X=X_batch,
                reference_stats=reference_stats,
                noise_scale=args.drift_noise,
                scale_factor=args.drift_scale,
                mean_shift=args.drift_mean_shift,
                rng=rng,
            )
            X_batch, y_batch, groups_batch = apply_class_prior_shift(
                X=X_batch,
                y=y_batch,
                groups=groups_batch,
                strength=args.class_prior_strength,
                rng=rng,
            )

        batch_df = pd.DataFrame(X_batch, columns=bundle.feature_names)
        drift_score, _ = aggregate_batch_drift(reference_df, batch_df, key_features, bins=args.bins)
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
        confidences = extract_softmax_confidences(batch_result["probabilities"])
        anomaly_scores = compute_anomaly_scores(X_batch, reference_stats)

        low_conf_mask = low_confidence_flags(confidences, threshold=args.confidence_threshold)
        suspicious_mask = suspicious_high_confidence_on_anomalies(
            anomaly_scores=anomaly_scores,
            confidences=confidences,
            anomaly_threshold=args.anomaly_threshold,
            confidence_threshold=args.high_confidence_threshold,
        )

        batch_candidates = select_hitl_candidates(
            batch_df=batch_df,
            y_true=batch_result["y_true"],
            y_pred=batch_result["y_pred"],
            confidences=confidences,
            anomaly_scores=anomaly_scores,
            drift_score=drift_score,
            args=args,
        )
        if not batch_candidates.empty:
            batch_candidates["batch_index"] = int(batch_index)
            batch_candidates["participants"] = ",".join(sorted(set(groups_batch.astype(str).tolist())))
            candidate_frames.append(batch_candidates)

            feedback_df = simulate_human_feedback(
                candidate_df=batch_candidates,
                annotator_accuracy=args.annotator_accuracy,
                labels=bundle.labels,
                rng=rng,
            )
            feedback_df["batch_index"] = int(batch_index)
            feedback_df["participants"] = ",".join(sorted(set(groups_batch.astype(str).tolist())))
            feedback_frames.append(feedback_df)
            total_reviewed += int(len(feedback_df))
            total_corrected += int(feedback_df["model_corrected"].sum())

        low_conf_fraction = float(np.mean(low_conf_mask)) if len(low_conf_mask) else 0.0
        intervention_triggered = bool(
            drift_score >= float(args.drift_trigger_threshold)
            or low_conf_fraction >= float(args.low_conf_fraction_trigger)
            or bool(np.any(suspicious_mask))
        )

        intervention_rows.append(
            {
                "batch_index": int(batch_index),
                "n_samples": int(len(y_batch)),
                "participants": ",".join(sorted(set(groups_batch.astype(str).tolist()))),
                "drift_applied": int(drift_applied),
                "drift_score": float(drift_score),
                "drift_threshold": float(args.drift_trigger_threshold),
                "low_conf_fraction": low_conf_fraction,
                "low_conf_fraction_trigger": float(args.low_conf_fraction_trigger),
                "suspicious_count": int(np.sum(suspicious_mask)),
                "selected_for_review": int(len(batch_candidates)),
                "intervention_triggered": int(intervention_triggered),
                "mean_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
            }
        )

    intervention_df = pd.DataFrame(intervention_rows)
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    feedback_df = pd.concat(feedback_frames, ignore_index=True) if feedback_frames else pd.DataFrame()

    if not feedback_df.empty:
        reviewed_accuracy = float(np.mean(feedback_df["reviewed_label"] == feedback_df["y_true"]))
        agreement_rate = float(np.mean(feedback_df["human_agrees_with_model"]))
        correction_rate = float(np.mean(feedback_df["model_corrected"]))
    else:
        reviewed_accuracy = 0.0
        agreement_rate = 0.0
        correction_rate = 0.0

    summary = {
        "source_model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": vars(args),
        "batches_processed": int(len(intervention_df)),
        "intervention_batches": int(intervention_df["intervention_triggered"].sum()) if not intervention_df.empty else 0,
        "samples_reviewed": int(total_reviewed),
        "samples_corrected": int(total_corrected),
        "review_coverage": float(total_reviewed / max(1, len(X_stream))),
        "human_review_accuracy": reviewed_accuracy,
        "human_model_agreement_rate": agreement_rate,
        "human_correction_rate": correction_rate,
        "mean_selected_per_triggered_batch": (
            float(intervention_df.loc[intervention_df["intervention_triggered"] == 1, "selected_for_review"].mean())
            if not intervention_df.empty and int(intervention_df["intervention_triggered"].sum()) > 0
            else 0.0
        ),
    }

    save_csv(output_dir / "intervention_log.csv", intervention_df)
    save_csv(output_dir / "hitl_candidates.csv", candidate_df)
    save_csv(output_dir / "human_feedback_log.csv", feedback_df)
    save_json(output_dir / "review_summary.json", summary)
    plot_hitl_timeline(intervention_df, output_dir / "hitl_timeline.png")

    (output_dir / "acceptance_notes.txt").write_text(
        "\n".join(
            [
                "Phase 3 acceptance checklist",
                f"- Batches processed: {len(intervention_df)}",
                f"- Intervention batches: {int(intervention_df['intervention_triggered'].sum()) if not intervention_df.empty else 0}",
                f"- Samples reviewed: {total_reviewed}",
                f"- Samples corrected: {total_corrected}",
                f"- Review coverage: {float(total_reviewed / max(1, len(X_stream))):.4f}",
                f"- Human feedback log: {output_dir / 'human_feedback_log.csv'}",
                f"- HITL candidate log: {output_dir / 'hitl_candidates.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "intervention_batches": int(intervention_df["intervention_triggered"].sum()) if not intervention_df.empty else 0,
                "samples_reviewed": total_reviewed,
                "samples_corrected": total_corrected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

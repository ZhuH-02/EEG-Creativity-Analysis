from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from active_learning_eval import select_query_indices
from adaptation_eval import (
    apply_class_prior_shift,
    build_replay_mix,
    fine_tune_model,
    simulate_feature_drift,
    split_adaptation_pool,
)
from analysis_utils import (
    evaluate_model,
    extract_softmax_confidences,
    load_dataset_for_run,
    load_trained_artifacts,
    make_output_dir,
    profile_inference_if_available,
    save_csv,
    save_json,
    select_key_features,
    set_seed,
    summarize_feature_reference,
)
from continual_learning import ReplayBuffer, current_rss_mb, save_model_version
from hitl_active_learning import select_hitl_candidates, simulate_human_feedback
from monitoring_sim import aggregate_batch_drift, batched_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4 Phase 5: end-to-end system pipeline with monitoring, HITL, active learning, and continual updates."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/milestone4")
    parser.add_argument("--run_name", type=str, default="phase5_system_pipeline")
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
    parser.add_argument("--low_conf_fraction_trigger", type=float, default=0.35)
    parser.add_argument("--confidence_threshold", type=float, default=0.55)
    parser.add_argument("--high_confidence_threshold", type=float, default=0.85)
    parser.add_argument("--anomaly_threshold", type=float, default=4.0)
    parser.add_argument("--active_strategy", type=str, default="uncertainty", choices=["random", "uncertainty", "hybrid"])
    parser.add_argument("--query_budget_per_trigger", type=int, default=16)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--annotator_accuracy", type=float, default=1.0)
    parser.add_argument("--annotation_time_sec_per_sample", type=float, default=2.5)
    parser.add_argument("--min_reviewed_for_update", type=int, default=48)
    parser.add_argument("--max_updates", type=int, default=3)
    parser.add_argument("--buffer_capacity", type=int, default=2048)
    parser.add_argument("--buffer_init_per_class", type=int, default=128)
    parser.add_argument("--clean_replay_ratio", type=float, default=0.5)
    parser.add_argument("--adaptation_split", type=float, default=0.8)
    parser.add_argument("--adaptation_epochs", type=int, default=8)
    parser.add_argument("--adaptation_patience", type=int, default=3)
    parser.add_argument("--adaptation_lr", type=float, default=5e-4)
    return parser.parse_args()


def plot_system_timeline(system_df: pd.DataFrame, rollout_df: pd.DataFrame, output_path: Path) -> None:
    if system_df.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    x = system_df["batch_index"].to_numpy(dtype=np.int64)

    axes[0].plot(x, system_df["drift_score"], marker="o", color="#bb3e03", label="Drift score")
    axes[0].axhline(float(system_df["drift_threshold"].iloc[0]), linestyle="--", color="#9b2226", alpha=0.7, label="Drift threshold")
    axes[0].set_ylabel("Drift score")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, system_df["query_count"], color="#0a9396", label="Queried")
    axes[1].plot(x, system_df["pending_reviewed_samples"], marker="o", color="#005f73", label="Pending reviewed")
    axes[1].axhline(float(system_df["min_reviewed_for_update"].iloc[0]), linestyle="--", color="#94d2bd", alpha=0.7, label="Update threshold")
    axes[1].set_ylabel("Counts")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].step(x, system_df["model_version"], where="mid", color="#6c584c", label="Model version")
    axes[2].plot(x, system_df["batch_macro_f1"], marker="o", color="#588157", label="Batch macro-F1")
    axes[2].set_xlabel("Batch index")
    axes[2].set_ylabel("Version / F1")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    for _, row in rollout_df.iterrows():
        for ax in axes:
            ax.axvline(float(row["trigger_batch"]), color="#ae2012", linestyle="--", alpha=0.4)

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
    X_clean_eval, y_clean_eval, _ = dataset.subset("test")

    reference_stats = summarize_feature_reference(X_reference, bundle.feature_names)
    reference_df = pd.DataFrame(X_reference, columns=bundle.feature_names)
    key_features = select_key_features(X_reference, bundle.feature_names, top_k=args.top_k_features)
    stream_batches = batched_indices(len(X_stream), args.stream_batch_size)

    baseline_clean = evaluate_model(
        model=bundle.model,
        X=X_clean_eval,
        y=y_clean_eval,
        device=bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )

    current_model = copy.deepcopy(bundle.model)
    current_bundle = replace(bundle, model=current_model)
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, labels=bundle.labels, rng=rng)
    replay_buffer.initialize(X_reference, y_reference, per_class_quota=args.buffer_init_per_class)

    reviewed_X_pending: List[np.ndarray] = []
    reviewed_y_pending: List[np.ndarray] = []
    reviewed_batch_ids: List[int] = []

    system_rows: List[Dict[str, Any]] = []
    trigger_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    rollout_rows: List[Dict[str, Any]] = []
    version_rows: List[Dict[str, Any]] = [
        {
            "version": 0,
            "type": "initial",
            "path": str(output_dir / "model_step_000.pt"),
            "parent_version": "",
            "trigger_batch": "",
            "trigger_reason": "",
        }
    ]
    save_model_version(output_dir / "model_step_000.pt", current_bundle, current_model, version=0)

    model_version = 0
    update_count = 0
    total_reviewed = 0
    total_estimated_human_time = 0.0

    drifted_batches_X: List[np.ndarray] = []
    drifted_batches_y: List[np.ndarray] = []

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

        drifted_batches_X.append(X_batch)
        drifted_batches_y.append(y_batch)

        batch_frame = pd.DataFrame(X_batch, columns=bundle.feature_names)
        drift_score, _ = aggregate_batch_drift(reference_df, batch_frame, key_features, bins=args.bins)
        batch_result = evaluate_model(
            model=current_bundle.model,
            X=X_batch,
            y=y_batch,
            device=current_bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=current_bundle.scaler,
            labels=current_bundle.labels,
            class_names=current_bundle.class_names,
        )
        confidences = extract_softmax_confidences(batch_result["probabilities"])
        low_conf_fraction = float(np.mean(confidences < float(args.confidence_threshold))) if len(confidences) else 0.0
        anomaly_scores = np.max(np.abs((X_batch - reference_stats["mean"].to_numpy(dtype=np.float64)) / np.maximum(reference_stats["std"].to_numpy(dtype=np.float64), 1e-8)), axis=1)

        intervention_triggered = bool(
            drift_score >= float(args.drift_trigger_threshold)
            or low_conf_fraction >= float(args.low_conf_fraction_trigger)
        )
        query_count = 0
        update_triggered = False
        trigger_reason = ""

        if intervention_triggered:
            trigger_reasons: List[str] = []
            if drift_score >= float(args.drift_trigger_threshold):
                trigger_reasons.append("drift")
            if low_conf_fraction >= float(args.low_conf_fraction_trigger):
                trigger_reasons.append("low_conf_fraction")
            trigger_reason = ",".join(trigger_reasons)

            temp_args = argparse.Namespace(**vars(args))
            temp_args.query_budget_per_batch = int(args.query_budget_per_trigger * max(1, args.candidate_multiplier))
            candidates_df = select_hitl_candidates(
                batch_df=batch_frame,
                y_true=batch_result["y_true"],
                y_pred=batch_result["y_pred"],
                confidences=confidences,
                anomaly_scores=anomaly_scores,
                drift_score=drift_score,
                args=temp_args,
            )

            if not candidates_df.empty:
                candidate_local_idx = candidates_df["row_index_in_batch"].to_numpy(dtype=int)
                candidate_probs = batch_result["probabilities"][candidate_local_idx]
                chosen_local = select_query_indices(
                    strategy=args.active_strategy,
                    X_pool=X_batch[candidate_local_idx],
                    probabilities=candidate_probs,
                    unlabeled_indices=np.arange(len(candidate_local_idx), dtype=int),
                    query_budget=args.query_budget_per_trigger,
                    candidate_multiplier=args.candidate_multiplier,
                    rng=rng,
                )
                selected_df = candidates_df.iloc[chosen_local].reset_index(drop=True) if len(chosen_local) else pd.DataFrame()

                if not selected_df.empty:
                    feedback_df = simulate_human_feedback(
                        candidate_df=selected_df,
                        annotator_accuracy=args.annotator_accuracy,
                        labels=bundle.labels,
                        rng=rng,
                    )
                    reviewed_idx = feedback_df["row_index_in_batch"].to_numpy(dtype=int)
                    reviewed_X_pending.append(X_batch[reviewed_idx])
                    reviewed_y_pending.append(feedback_df["reviewed_label"].to_numpy(dtype=np.int64))
                    reviewed_batch_ids.append(batch_index)
                    query_count = int(len(feedback_df))
                    total_reviewed += query_count
                    total_estimated_human_time += float(query_count * float(args.annotation_time_sec_per_sample))

                    for row in feedback_df.to_dict(orient="records"):
                        query_rows.append(
                            {
                                "batch_index": int(batch_index),
                                "participants": ",".join(sorted(set(groups_batch.astype(str).tolist()))),
                                "strategy": args.active_strategy,
                                **row,
                            }
                        )

            trigger_rows.append(
                {
                    "batch_index": int(batch_index),
                    "event_type": "intervention",
                    "triggered": int(intervention_triggered),
                    "reason": trigger_reason,
                    "query_count": int(query_count),
                    "drift_score": float(drift_score),
                    "low_conf_fraction": float(low_conf_fraction),
                    "model_version_before": int(model_version),
                }
            )

        pending_reviewed_samples = int(sum(len(arr) for arr in reviewed_y_pending)) if reviewed_y_pending else 0
        if pending_reviewed_samples >= int(args.min_reviewed_for_update) and update_count < int(args.max_updates):
            X_reviewed = np.vstack(reviewed_X_pending)
            y_reviewed = np.concatenate(reviewed_y_pending)
            X_adapt_train, y_adapt_train, X_adapt_holdout, y_adapt_holdout = split_adaptation_pool(
                X_reviewed,
                y_reviewed,
                ratio=args.adaptation_split,
                rng=rng,
            )
            replay_take = int(round(len(X_adapt_train) * max(0.0, float(args.clean_replay_ratio))))
            X_replay, y_replay = replay_buffer.sample(replay_take)
            X_mix, y_mix = build_replay_mix(
                X_clean_train=X_replay,
                y_clean_train=y_replay,
                X_drift_adapt=X_adapt_train,
                y_drift_adapt=y_adapt_train,
                replay_ratio=args.clean_replay_ratio,
                rng=rng,
            )

            before_holdout = evaluate_model(
                model=current_bundle.model,
                X=X_adapt_holdout,
                y=y_adapt_holdout,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )
            update_start = time.perf_counter()
            peak_ram_before = current_rss_mb()
            adapted_model, adaptation_training = fine_tune_model(
                bundle=current_bundle,
                X_train=X_mix,
                y_train=y_mix,
                X_val=X_adapt_holdout,
                y_val=y_adapt_holdout,
                epochs=args.adaptation_epochs,
                patience=args.adaptation_patience,
                learning_rate=args.adaptation_lr,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            update_time_sec = time.perf_counter() - update_start
            peak_ram_after = current_rss_mb()

            after_holdout = evaluate_model(
                model=adapted_model,
                X=X_adapt_holdout,
                y=y_adapt_holdout,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )

            update_count += 1
            model_version += 1
            current_model = adapted_model
            current_bundle = replace(current_bundle, model=current_model)
            replay_buffer.add(X_adapt_train, y_adapt_train)

            model_path = output_dir / f"model_step_{model_version:03d}.pt"
            save_model_version(model_path, current_bundle, current_model, version=model_version)
            version_rows.append(
                {
                    "version": model_version,
                    "type": "continual_rollout",
                    "path": str(model_path),
                    "parent_version": model_version - 1,
                    "trigger_batch": batch_index,
                    "trigger_reason": "reviewed_label_threshold",
                }
            )
            rollout_rows.append(
                {
                    "update_index": int(update_count),
                    "model_version": int(model_version),
                    "trigger_batch": int(batch_index),
                    "trigger_reason": "reviewed_label_threshold",
                    "reviewed_samples_used": int(len(X_reviewed)),
                    "replay_samples_used": int(len(X_replay)),
                    "holdout_macro_f1_before": float(before_holdout["macro_f1"]),
                    "holdout_macro_f1_after": float(after_holdout["macro_f1"]),
                    "update_time_sec": float(update_time_sec),
                    "peak_ram_mb_before": peak_ram_before,
                    "peak_ram_mb_after": peak_ram_after,
                    "best_val_macro_f1": float(adaptation_training["best_val_macro_f1"]),
                }
            )
            trigger_rows.append(
                {
                    "batch_index": int(batch_index),
                    "event_type": "rollout",
                    "triggered": 1,
                    "reason": "reviewed_label_threshold",
                    "query_count": 0,
                    "drift_score": float(drift_score),
                    "low_conf_fraction": float(low_conf_fraction),
                    "model_version_before": int(model_version - 1),
                }
            )

            reviewed_X_pending = []
            reviewed_y_pending = []
            reviewed_batch_ids = []
            pending_reviewed_samples = 0
            update_triggered = True

        system_rows.append(
            {
                "batch_index": int(batch_index),
                "participants": ",".join(sorted(set(groups_batch.astype(str).tolist()))),
                "drift_applied": int(drift_applied),
                "drift_score": float(drift_score),
                "drift_threshold": float(args.drift_trigger_threshold),
                "low_conf_fraction": float(low_conf_fraction),
                "low_conf_fraction_trigger": float(args.low_conf_fraction_trigger),
                "batch_accuracy": float(batch_result["accuracy"]),
                "batch_macro_f1": float(batch_result["macro_f1"]),
                "mean_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
                "intervention_triggered": int(intervention_triggered),
                "query_count": int(query_count),
                "pending_reviewed_samples": int(pending_reviewed_samples),
                "min_reviewed_for_update": int(args.min_reviewed_for_update),
                "update_triggered": int(update_triggered),
                "model_version": int(model_version),
                "buffer_size": int(replay_buffer.size),
            }
        )

    X_drift_full = np.vstack(drifted_batches_X) if drifted_batches_X else np.zeros((0, X_reference.shape[1]), dtype=np.float64)
    y_drift_full = np.concatenate(drifted_batches_y) if drifted_batches_y else np.zeros((0,), dtype=np.int64)

    final_clean = evaluate_model(
        model=current_bundle.model,
        X=X_clean_eval,
        y=y_clean_eval,
        device=current_bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=current_bundle.scaler,
        labels=current_bundle.labels,
        class_names=current_bundle.class_names,
    )
    final_drift = evaluate_model(
        model=current_bundle.model,
        X=X_drift_full,
        y=y_drift_full,
        device=current_bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=current_bundle.scaler,
        labels=current_bundle.labels,
        class_names=current_bundle.class_names,
    )
    final_profile = profile_inference_if_available(
        current_bundle,
        X_drift_full,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    system_df = pd.DataFrame(system_rows)
    trigger_df = pd.DataFrame(trigger_rows)
    query_df = pd.DataFrame(query_rows)
    rollout_df = pd.DataFrame(rollout_rows)
    version_df = pd.DataFrame(version_rows)

    save_csv(output_dir / "system_run_log.csv", system_df)
    save_csv(output_dir / "trigger_log.csv", trigger_df)
    save_csv(output_dir / "query_feedback_log.csv", query_df)
    save_csv(output_dir / "rollout_history.csv", rollout_df)
    save_csv(output_dir / "model_versions.csv", version_df)
    save_json(output_dir / "rollout_history.json", version_rows)
    plot_system_timeline(system_df, rollout_df, output_dir / "system_timeline.png")

    summary = {
        "source_model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": vars(args),
        "baseline_clean": {
            "accuracy": float(baseline_clean["accuracy"]),
            "macro_f1": float(baseline_clean["macro_f1"]),
        },
        "final_clean": {
            "accuracy": float(final_clean["accuracy"]),
            "macro_f1": float(final_clean["macro_f1"]),
        },
        "final_drifted": {
            "accuracy": float(final_drift["accuracy"]),
            "macro_f1": float(final_drift["macro_f1"]),
        },
        "system_counts": {
            "batches_processed": int(len(system_df)),
            "intervention_batches": int(system_df["intervention_triggered"].sum()) if not system_df.empty else 0,
            "total_queries": int(len(query_df)),
            "updates_completed": int(len(rollout_df)),
            "final_model_version": int(model_version),
        },
        "resources": {
            "estimated_human_time_sec": float(total_estimated_human_time),
            "final_buffer_size": int(replay_buffer.size),
            "final_inference": final_profile,
        },
    }
    save_json(output_dir / "end_to_end_summary.json", summary)

    (output_dir / "acceptance_notes.txt").write_text(
        "\n".join(
            [
                "Phase 5 acceptance checklist",
                f"- Batches processed: {len(system_df)}",
                f"- Intervention batches: {int(system_df['intervention_triggered'].sum()) if not system_df.empty else 0}",
                f"- Queries issued: {len(query_df)}",
                f"- Rollouts completed: {len(rollout_df)}",
                f"- Final model version: {model_version}",
                f"- System run log: {output_dir / 'system_run_log.csv'}",
                f"- Rollout history: {output_dir / 'rollout_history.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "intervention_batches": int(system_df["intervention_triggered"].sum()) if not system_df.empty else 0,
                "queries": int(len(query_df)),
                "rollouts": int(len(rollout_df)),
                "final_model_version": int(model_version),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

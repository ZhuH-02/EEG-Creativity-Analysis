from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    import psutil
except Exception:
    psutil = None

from adaptation_eval import (
    apply_class_prior_shift,
    build_replay_mix,
    fine_tune_model,
    simulate_feature_drift,
    split_adaptation_pool,
)
from analysis_utils import (
    ArtifactBundle,
    compute_common_metrics,
    evaluate_model,
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
from monitoring_sim import aggregate_batch_drift, batched_indices


class ReplayBuffer:
    """Simple class-balanced replay buffer for continual updates."""

    def __init__(self, capacity: int, labels: List[int], rng: np.random.Generator):
        self.capacity = int(max(1, capacity))
        self.labels = [int(label) for label in labels]
        self.rng = rng
        self._X = np.zeros((0, 0), dtype=np.float64)
        self._y = np.zeros((0,), dtype=np.int64)

    @property
    def size(self) -> int:
        return int(len(self._y))

    def initialize(self, X: np.ndarray, y: np.ndarray, per_class_quota: int | None = None) -> None:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        if len(X_arr) == 0:
            self._X = np.zeros((0, 0), dtype=np.float64)
            self._y = np.zeros((0,), dtype=np.int64)
            return

        quota = int(per_class_quota or max(1, self.capacity // max(1, len(self.labels))))
        chosen_indices: List[int] = []
        for label in self.labels:
            label_idx = np.where(y_arr == int(label))[0]
            if len(label_idx) == 0:
                continue
            take = min(quota, len(label_idx))
            picked = self.rng.choice(label_idx, size=take, replace=False)
            chosen_indices.extend(int(v) for v in picked.tolist())

        if len(chosen_indices) > self.capacity:
            chosen_indices = self.rng.choice(np.array(chosen_indices, dtype=int), size=self.capacity, replace=False).tolist()

        chosen_indices = sorted(set(chosen_indices))
        self._X = X_arr[chosen_indices].copy()
        self._y = y_arr[chosen_indices].copy()
        self._rebalance_if_needed()

    def add(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        X_arr = np.asarray(X_new, dtype=np.float64)
        y_arr = np.asarray(y_new, dtype=np.int64)
        if len(X_arr) == 0:
            return

        if self.size == 0:
            self._X = X_arr.copy()
            self._y = y_arr.copy()
        else:
            self._X = np.vstack([self._X, X_arr])
            self._y = np.concatenate([self._y, y_arr])
        self._rebalance_if_needed()

    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0 or int(n_samples) <= 0:
            return np.zeros((0, self._X.shape[1] if self._X.ndim == 2 else 0), dtype=np.float64), np.zeros((0,), dtype=np.int64)

        take_total = min(int(n_samples), self.size)
        per_class = max(1, take_total // max(1, len(self.labels)))
        chosen_indices: List[int] = []

        for label in self.labels:
            label_idx = np.where(self._y == int(label))[0]
            if len(label_idx) == 0:
                continue
            take = min(per_class, len(label_idx))
            picked = self.rng.choice(label_idx, size=take, replace=False)
            chosen_indices.extend(int(v) for v in picked.tolist())

        remaining = take_total - len(chosen_indices)
        if remaining > 0:
            all_idx = np.arange(self.size, dtype=int)
            leftover = np.setdiff1d(all_idx, np.array(chosen_indices, dtype=int), assume_unique=False)
            if len(leftover) > 0:
                picked = self.rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
                chosen_indices.extend(int(v) for v in picked.tolist())

        chosen_indices = sorted(set(chosen_indices))
        return self._X[chosen_indices].copy(), self._y[chosen_indices].copy()

    def stats_row(self, stage: str, step: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "stage": stage,
            "step": int(step),
            "buffer_size": self.size,
            "capacity": self.capacity,
            "utilization": float(self.size / max(1, self.capacity)),
        }
        for label in self.labels:
            row[f"class_{label}_count"] = int(np.sum(self._y == int(label)))
        return row

    def _rebalance_if_needed(self) -> None:
        if self.size <= self.capacity:
            return

        keep_quota = max(1, self.capacity // max(1, len(self.labels)))
        keep_indices: List[int] = []
        for label in self.labels:
            label_idx = np.where(self._y == int(label))[0]
            if len(label_idx) == 0:
                continue
            take = min(keep_quota, len(label_idx))
            picked = self.rng.choice(label_idx, size=take, replace=False)
            keep_indices.extend(int(v) for v in picked.tolist())

        remaining = self.capacity - len(keep_indices)
        if remaining > 0:
            all_idx = np.arange(self.size, dtype=int)
            leftover = np.setdiff1d(all_idx, np.array(keep_indices, dtype=int), assume_unique=False)
            if len(leftover) > 0:
                picked = self.rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
                keep_indices.extend(int(v) for v in picked.tolist())

        keep_indices = sorted(set(keep_indices))[: self.capacity]
        self._X = self._X[keep_indices].copy()
        self._y = self._y[keep_indices].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4 Phase 1: minimal continual learning loop with drift-triggered replay updates."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/milestone4")
    parser.add_argument("--run_name", type=str, default="continual_eval")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reference_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--stream_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--stream_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling_window_batches", type=int, default=3)
    parser.add_argument("--top_k_features", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--drift_start_batch", type=int, default=4)
    parser.add_argument("--drift_noise", type=float, default=0.05)
    parser.add_argument("--drift_scale", type=float, default=1.2)
    parser.add_argument("--drift_mean_shift", type=float, default=0.25)
    parser.add_argument("--class_prior_strength", type=float, default=0.0)
    parser.add_argument("--drift_trigger_threshold", type=float, default=1.5)
    parser.add_argument("--macro_f1_drop_threshold", type=float, default=0.08)
    parser.add_argument("--min_drift_batches_for_update", type=int, default=2)
    parser.add_argument("--max_updates", type=int, default=2)
    parser.add_argument("--adaptation_split", type=float, default=0.8)
    parser.add_argument("--replay_ratio", type=float, default=0.5)
    parser.add_argument("--buffer_capacity", type=int, default=2048)
    parser.add_argument("--buffer_init_per_class", type=int, default=128)
    parser.add_argument("--adaptation_epochs", type=int, default=8)
    parser.add_argument("--adaptation_patience", type=int, default=3)
    parser.add_argument("--adaptation_lr", type=float, default=5e-4)
    return parser.parse_args()


def current_rss_mb() -> float | None:
    if psutil is None:
        return None
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def save_model_version(path: Path, bundle: ArtifactBundle, model: torch.nn.Module, version: int) -> None:
    payload = {
        "model_name": bundle.model_name,
        "model_state_dict": model.state_dict(),
        "scaler_mean": bundle.scaler.mean.tolist() if bundle.scaler.mean is not None else None,
        "scaler_std": bundle.scaler.std.tolist() if bundle.scaler.std is not None else None,
        "feature_names": bundle.feature_names,
        "labels": bundle.labels,
        "class_names": bundle.class_names,
        "source_model_dir": str(bundle.model_dir),
        "continual_version": int(version),
    }
    torch.save(payload, path)


def rolling_slice(history: List[np.ndarray], window: int) -> np.ndarray:
    tail = history[-int(window) :] if int(window) > 0 else history
    return np.concatenate(tail, axis=0) if tail else np.zeros((0,), dtype=np.int64)


def evaluate_bundle(
    bundle: ArtifactBundle,
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    temp_bundle = replace(bundle, model=model)
    return evaluate_model(
        model=temp_bundle.model,
        X=X,
        y=y,
        device=temp_bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=temp_bundle.scaler,
        labels=temp_bundle.labels,
        class_names=temp_bundle.class_names,
    )


def plot_metric_trajectory(batch_df: pd.DataFrame, update_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = batch_df["batch_index"].to_numpy(dtype=np.int64)

    axes[0].plot(x, batch_df["batch_macro_f1"], marker="o", label="Batch macro-F1", color="#0a9396")
    axes[0].plot(x, batch_df["rolling_macro_f1"], marker="o", label="Rolling macro-F1", color="#005f73")
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("Continual Learning Metric Trajectory")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, batch_df["drift_score"], marker="o", label="Drift score", color="#bb3e03")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("Batch index")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for axis in axes:
        for _, row in update_df.iterrows():
            axis.axvline(float(row["trigger_batch"]), color="#ae2012", linestyle="--", alpha=0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_update_effects(update_df: pd.DataFrame, output_path: Path) -> None:
    if update_df.empty:
        return

    labels = [f"U{int(v)}" for v in update_df["update_index"].tolist()]
    before_values = update_df["holdout_macro_f1_before"].to_numpy(dtype=np.float64)
    after_values = update_df["holdout_macro_f1_after"].to_numpy(dtype=np.float64)
    positions = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(positions - width / 2.0, before_values, width=width, label="Before", color="#94d2bd")
    ax.bar(positions + width / 2.0, after_values, width=width, label="After", color="#0a9396")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Holdout macro-F1")
    ax.set_title("Per-update Continual Learning Effect")
    ax.grid(alpha=0.25)
    ax.legend()
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

    initial_result = evaluate_bundle(
        bundle=bundle,
        model=bundle.model,
        X=X_stream,
        y=y_stream,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    baseline_macro_f1 = float(initial_result["macro_f1"])
    baseline_accuracy = float(initial_result["accuracy"])

    current_model = bundle.model
    current_bundle = bundle
    model_version = 0
    update_count = 0

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity, labels=bundle.labels, rng=rng)
    replay_buffer.initialize(X_reference, y_reference, per_class_quota=args.buffer_init_per_class)

    batch_rows: List[Dict[str, Any]] = []
    update_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    buffer_rows: List[Dict[str, Any]] = [replay_buffer.stats_row(stage="init", step=0)]
    version_rows: List[Dict[str, Any]] = [
        {
            "version": 0,
            "type": "initial",
            "path": str(output_dir / "model_step_000.pt"),
            "parent_version": "",
            "trigger_batch": "",
            "trigger_reasons": "",
        }
    ]
    save_model_version(output_dir / "model_step_000.pt", current_bundle, current_model, version=0)

    y_history: List[np.ndarray] = []
    pred_history: List[np.ndarray] = []
    prob_history: List[np.ndarray] = []
    pending_X: List[np.ndarray] = []
    pending_y: List[np.ndarray] = []
    pending_groups: List[np.ndarray] = []
    pending_batch_ids: List[int] = []

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
        drift_score, feature_rows = aggregate_batch_drift(reference_df, batch_df, key_features, bins=args.bins)
        for feature_row in feature_rows:
            drift_rows.append({"batch_index": batch_index, **feature_row})

        batch_result = evaluate_bundle(
            bundle=current_bundle,
            model=current_model,
            X=X_batch,
            y=y_batch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        y_history.append(np.asarray(batch_result["y_true"], dtype=np.int64))
        pred_history.append(np.asarray(batch_result["y_pred"], dtype=np.int64))
        prob_history.append(np.asarray(batch_result["probabilities"], dtype=np.float64))

        rolling_metrics = compute_common_metrics(
            y_true=rolling_slice(y_history, args.rolling_window_batches),
            y_pred=rolling_slice(pred_history, args.rolling_window_batches),
            probabilities=np.concatenate(prob_history[-int(args.rolling_window_batches) :], axis=0)
            if prob_history
            else np.zeros((0, 0), dtype=np.float64),
            labels=current_bundle.labels,
            class_names=current_bundle.class_names,
        )

        trigger_reasons: List[str] = []
        if drift_applied and drift_score >= float(args.drift_trigger_threshold):
            trigger_reasons.append("drift_score")
        if drift_applied and float(rolling_metrics["macro_f1"]) <= baseline_macro_f1 - float(args.macro_f1_drop_threshold):
            trigger_reasons.append("macro_f1_drop")

        update_triggered = False
        replay_sample_size = int(round(len(X_batch) * max(0.0, float(args.replay_ratio))))
        if drift_applied:
            pending_X.append(X_batch)
            pending_y.append(y_batch)
            pending_groups.append(groups_batch)
            pending_batch_ids.append(batch_index)

        decision_rows.append(
            {
                "batch_index": batch_index,
                "drift_applied": int(drift_applied),
                "drift_score": float(drift_score),
                "rolling_macro_f1": float(rolling_metrics["macro_f1"]),
                "baseline_macro_f1": float(baseline_macro_f1),
                "drift_threshold": float(args.drift_trigger_threshold),
                "macro_f1_drop_threshold": float(args.macro_f1_drop_threshold),
                "pending_batches": len(pending_batch_ids),
                "pending_samples": int(sum(len(arr) for arr in pending_y)) if pending_y else 0,
                "buffer_size": replay_buffer.size,
                "candidate_reasons": ",".join(trigger_reasons),
                "update_triggered": 0,
                "model_version_before": int(model_version),
            }
        )

        if (
            trigger_reasons
            and len(pending_X) >= int(args.min_drift_batches_for_update)
            and update_count < int(args.max_updates)
        ):
            X_pending = np.vstack(pending_X)
            y_pending = np.concatenate(pending_y)
            groups_pending = np.concatenate(pending_groups)

            X_adapt_train, y_adapt_train, X_adapt_holdout, y_adapt_holdout = split_adaptation_pool(
                X_pending,
                y_pending,
                ratio=args.adaptation_split,
                rng=rng,
            )
            before_holdout = evaluate_bundle(
                bundle=current_bundle,
                model=current_model,
                X=X_adapt_holdout,
                y=y_adapt_holdout,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            X_replay, y_replay = replay_buffer.sample(replay_sample_size)
            X_mix, y_mix = build_replay_mix(
                X_clean_train=X_replay,
                y_clean_train=y_replay,
                X_drift_adapt=X_adapt_train,
                y_drift_adapt=y_adapt_train,
                replay_ratio=args.replay_ratio,
                rng=rng,
            )

            update_peak_ram_before = current_rss_mb()
            update_start = time.perf_counter()
            adapted_model, adaptation_training = fine_tune_model(
                bundle=replace(current_bundle, model=current_model),
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
            update_peak_ram_after = current_rss_mb()

            after_holdout = evaluate_bundle(
                bundle=current_bundle,
                model=adapted_model,
                X=X_adapt_holdout,
                y=y_adapt_holdout,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            update_profile = profile_inference_if_available(
                replace(current_bundle, model=adapted_model),
                X=X_adapt_holdout,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            update_count += 1
            model_version += 1
            current_model = adapted_model
            current_bundle = replace(current_bundle, model=current_model)
            buffer_size_before_update = replay_buffer.size
            replay_buffer.add(X_adapt_train, y_adapt_train)
            buffer_rows.append(replay_buffer.stats_row(stage="post_update", step=model_version))

            model_path = output_dir / f"model_step_{model_version:03d}.pt"
            save_model_version(model_path, current_bundle, current_model, version=model_version)
            version_rows.append(
                {
                    "version": model_version,
                    "type": "continual_update",
                    "path": str(model_path),
                    "parent_version": model_version - 1,
                    "trigger_batch": batch_index,
                    "trigger_reasons": ",".join(trigger_reasons),
                }
            )

            update_rows.append(
                {
                    "update_index": update_count,
                    "model_version": model_version,
                    "trigger_batch": batch_index,
                    "trigger_reasons": ",".join(trigger_reasons),
                    "pending_batches": ",".join(str(v) for v in pending_batch_ids),
                    "pending_samples": int(len(X_pending)),
                    "holdout_samples": int(len(X_adapt_holdout)),
                    "replay_sample_size": int(len(X_replay)),
                    "buffer_size_before_update": int(buffer_size_before_update),
                    "buffer_size_after_update": int(replay_buffer.size),
                    "holdout_macro_f1_before": float(before_holdout["macro_f1"]),
                    "holdout_macro_f1_after": float(after_holdout["macro_f1"]),
                    "holdout_accuracy_before": float(before_holdout["accuracy"]),
                    "holdout_accuracy_after": float(after_holdout["accuracy"]),
                    "update_time_sec": float(update_time_sec),
                    "peak_ram_mb_before": update_peak_ram_before,
                    "peak_ram_mb_after": update_peak_ram_after,
                    "inference_latency_p50_ms_after": update_profile.get("latency_per_sample_ms_p50"),
                    "inference_throughput_after": update_profile.get("throughput_samples_per_sec"),
                    "best_val_macro_f1": float(adaptation_training["best_val_macro_f1"]),
                }
            )

            pending_X = []
            pending_y = []
            pending_groups = []
            pending_batch_ids = []
            update_triggered = True
            decision_rows[-1]["update_triggered"] = 1
            decision_rows[-1]["model_version_after"] = int(model_version)
        else:
            decision_rows[-1]["model_version_after"] = int(model_version)

        batch_rows.append(
            {
                "batch_index": batch_index,
                "n_samples": int(len(y_batch)),
                "participants": ",".join(sorted(set(groups_batch.astype(str).tolist()))),
                "drift_applied": int(drift_applied),
                "drift_score": float(drift_score),
                "batch_accuracy": float(batch_result["accuracy"]),
                "batch_macro_f1": float(batch_result["macro_f1"]),
                "rolling_accuracy": float(rolling_metrics["accuracy"]),
                "rolling_macro_f1": float(rolling_metrics["macro_f1"]),
                "mean_confidence": float(np.mean(batch_result["confidences"])) if len(batch_result["confidences"]) else 0.0,
                "buffer_size": int(replay_buffer.size),
                "pending_samples": int(sum(len(arr) for arr in pending_y)) if pending_y else 0,
                "model_version": int(model_version),
                "update_triggered": int(update_triggered),
                "trigger_reasons": ",".join(trigger_reasons),
            }
        )

    final_result = evaluate_bundle(
        bundle=current_bundle,
        model=current_model,
        X=X_stream,
        y=y_stream,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    final_profile = profile_inference_if_available(
        replace(current_bundle, model=current_model),
        X=X_stream,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    batch_df = pd.DataFrame(batch_rows)
    update_df = pd.DataFrame(update_rows)
    decision_df = pd.DataFrame(decision_rows)
    drift_df = pd.DataFrame(drift_rows)
    buffer_df = pd.DataFrame(buffer_rows)
    versions_df = pd.DataFrame(version_rows)

    save_csv(output_dir / "batch_metrics.csv", batch_df)
    save_csv(output_dir / "update_history.csv", update_df)
    save_csv(output_dir / "update_decisions.csv", decision_df)
    save_csv(output_dir / "drift_metrics.csv", drift_df)
    save_csv(output_dir / "buffer_stats.csv", buffer_df)
    save_csv(output_dir / "model_versions.csv", versions_df)
    save_json(output_dir / "version_history.json", version_rows)

    plot_metric_trajectory(batch_df, update_df, output_dir / "metric_trajectory.png")
    plot_update_effects(update_df, output_dir / "update_effects.png")

    summary = {
        "source_model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": vars(args),
        "baseline_stream_metrics": {
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
        },
        "final_stream_metrics": {
            "accuracy": float(final_result["accuracy"]),
            "macro_f1": float(final_result["macro_f1"]),
            "weighted_f1": float(final_result["weighted_f1"]),
        },
        "updates_completed": int(update_count),
        "final_model_version": int(model_version),
        "buffer": {
            "capacity": int(args.buffer_capacity),
            "initial_per_class": int(args.buffer_init_per_class),
            "final_size": int(replay_buffer.size),
        },
        "mean_batch_macro_f1": float(batch_df["batch_macro_f1"].mean()) if not batch_df.empty else 0.0,
        "mean_drift_score": float(batch_df["drift_score"].mean()) if not batch_df.empty else 0.0,
        "max_drift_score": float(batch_df["drift_score"].max()) if not batch_df.empty else 0.0,
        "efficiency": {
            "final_inference": final_profile,
            "total_update_time_sec": float(update_df["update_time_sec"].sum()) if not update_df.empty else 0.0,
        },
    }
    save_json(output_dir / "continual_summary.json", summary)

    (output_dir / "acceptance_notes.txt").write_text(
        "\n".join(
            [
                "Phase 2 acceptance checklist",
                f"- Drift batches processed: {int(batch_df['drift_applied'].sum()) if not batch_df.empty else 0}",
                f"- Continual updates completed: {update_count}",
                f"- Baseline stream macro-F1: {baseline_macro_f1:.4f}",
                f"- Final stream macro-F1: {float(final_result['macro_f1']):.4f}",
                f"- Replay buffer final size: {replay_buffer.size}",
                f"- Metric trajectory plot: {output_dir / 'metric_trajectory.png'}",
                f"- Update history table: {output_dir / 'update_history.csv'}",
                f"- Update decision log: {output_dir / 'update_decisions.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "updates_completed": update_count,
                "baseline_macro_f1": baseline_macro_f1,
                "final_macro_f1": float(final_result["macro_f1"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

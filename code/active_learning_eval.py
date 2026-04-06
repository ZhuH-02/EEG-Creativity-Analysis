from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    set_seed,
    summarize_feature_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 4 Phase 4: active learning strategies with human feedback simulation."
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/milestone4")
    parser.add_argument("--run_name", type=str, default="phase4_active_learning")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift_noise", type=float, default=0.05)
    parser.add_argument("--drift_scale", type=float, default=1.2)
    parser.add_argument("--drift_mean_shift", type=float, default=0.25)
    parser.add_argument("--class_prior_strength", type=float, default=0.0)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--query_budget", type=int, default=32)
    parser.add_argument("--candidate_multiplier", type=int, default=4)
    parser.add_argument("--clean_replay_ratio", type=float, default=0.5)
    parser.add_argument("--adaptation_split", type=float, default=0.8)
    parser.add_argument("--adaptation_epochs", type=int, default=8)
    parser.add_argument("--adaptation_patience", type=int, default=3)
    parser.add_argument("--adaptation_lr", type=float, default=5e-4)
    parser.add_argument("--annotator_accuracy", type=float, default=1.0)
    parser.add_argument(
        "--strategies",
        type=str,
        default="random,uncertainty,hybrid",
        help="Comma-separated subset of: random, uncertainty, hybrid",
    )
    return parser.parse_args()


def parse_strategies(raw: str) -> List[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    allowed = {"random", "uncertainty", "hybrid"}
    parsed = [value for value in values if value in allowed]
    if not parsed:
        return ["random", "uncertainty", "hybrid"]
    return parsed


def softmax_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def margin_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape[1] < 2:
        return np.ones((len(probs),), dtype=np.float64)
    sorted_probs = np.sort(probs, axis=1)
    return 1.0 - (sorted_probs[:, -1] - sorted_probs[:, -2])


def simulate_human_oracle(y_true: np.ndarray, labels: Sequence[int], accuracy: float, rng: np.random.Generator) -> np.ndarray:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    label_list = [int(v) for v in labels]
    outputs = y_true_arr.copy()
    for idx in range(len(outputs)):
        if rng.random() <= float(accuracy):
            continue
        alternatives = [label for label in label_list if label != int(y_true_arr[idx])]
        if alternatives:
            outputs[idx] = int(rng.choice(alternatives))
    return outputs


def select_query_indices(
    strategy: str,
    X_pool: np.ndarray,
    probabilities: np.ndarray,
    unlabeled_indices: np.ndarray,
    query_budget: int,
    candidate_multiplier: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(unlabeled_indices) == 0:
        return np.zeros((0,), dtype=int)

    take = min(int(query_budget), len(unlabeled_indices))
    if strategy == "random":
        return np.asarray(rng.choice(unlabeled_indices, size=take, replace=False), dtype=int)

    entropy_scores = softmax_entropy(probabilities)
    margin_scores = margin_uncertainty(probabilities)
    uncertainty_score = entropy_scores + margin_scores
    ranked_local = np.argsort(-uncertainty_score, kind="mergesort")

    if strategy == "uncertainty":
        chosen_local = ranked_local[:take]
        return unlabeled_indices[chosen_local].astype(int)

    candidate_count = min(len(unlabeled_indices), max(take, int(take * max(1, candidate_multiplier))))
    candidate_local = ranked_local[:candidate_count]
    candidate_global = unlabeled_indices[candidate_local]
    candidate_X = np.asarray(X_pool[candidate_global], dtype=np.float64)

    selected_positions: List[int] = []
    remaining_positions = list(range(len(candidate_global)))
    if not remaining_positions:
        return np.zeros((0,), dtype=int)

    # Seed with most uncertain sample.
    selected_positions.append(remaining_positions.pop(0))
    while remaining_positions and len(selected_positions) < take:
        best_pos = remaining_positions[0]
        best_score = -np.inf
        selected_X = candidate_X[selected_positions]
        for pos in remaining_positions:
            point = candidate_X[pos]
            distances = np.linalg.norm(selected_X - point, axis=1)
            diversity = float(np.min(distances)) if len(distances) else 0.0
            score = float(uncertainty_score[candidate_local[pos]]) + 0.25 * diversity
            if score > best_score:
                best_score = score
                best_pos = pos
        selected_positions.append(best_pos)
        remaining_positions.remove(best_pos)

    return candidate_global[selected_positions].astype(int)


def plot_active_learning_curve(cycle_df: pd.DataFrame, output_path: Path) -> None:
    if cycle_df.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for strategy, group in cycle_df.groupby("strategy"):
        ordered = group.sort_values("cycle_index")
        axes[0].plot(
            ordered["cumulative_labels"],
            ordered["drifted_macro_f1_after"],
            marker="o",
            linewidth=2.0,
            label=strategy,
        )
        axes[1].plot(
            ordered["cumulative_labels"],
            ordered["update_time_sec"],
            marker="o",
            linewidth=2.0,
            label=strategy,
        )

    axes[0].set_ylabel("Drifted macro-F1 after update")
    axes[0].set_title("Active Learning Performance Across Labeling Cycles")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("Cumulative reviewed labels")
    axes[1].set_ylabel("Update time (sec)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    strategies = parse_strategies(args.strategies)
    set_seed(args.seed)
    master_rng = np.random.default_rng(args.seed)
    output_dir = make_output_dir(args.output_dir, args.run_name)

    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_train, y_train, _ = dataset.subset("train")
    X_val, y_val, groups_val = dataset.subset("val")
    X_test, y_test, groups_test = dataset.subset("test")

    reference_stats = summarize_feature_reference(X_train, bundle.feature_names)
    X_pool = simulate_feature_drift(
        X_val,
        reference_stats=reference_stats,
        noise_scale=args.drift_noise,
        scale_factor=args.drift_scale,
        mean_shift=args.drift_mean_shift,
        rng=master_rng,
    )
    X_eval_drift = simulate_feature_drift(
        X_test,
        reference_stats=reference_stats,
        noise_scale=args.drift_noise,
        scale_factor=args.drift_scale,
        mean_shift=args.drift_mean_shift,
        rng=master_rng,
    )
    X_pool, y_pool, groups_pool = apply_class_prior_shift(
        X_pool,
        y_val,
        groups_val,
        strength=args.class_prior_strength,
        rng=master_rng,
    )
    X_eval_drift, y_eval_drift, groups_eval_drift = apply_class_prior_shift(
        X_eval_drift,
        y_test,
        groups_test,
        strength=args.class_prior_strength,
        rng=master_rng,
    )

    base_clean = evaluate_model(
        model=bundle.model,
        X=X_test,
        y=y_test,
        device=bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )
    base_drift = evaluate_model(
        model=bundle.model,
        X=X_eval_drift,
        y=y_eval_drift,
        device=bundle.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )

    cycle_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    efficiency_rows: List[Dict[str, Any]] = []

    for strategy_idx, strategy in enumerate(strategies):
        strategy_rng = np.random.default_rng(args.seed + (strategy_idx + 1) * 100)
        current_model = copy.deepcopy(bundle.model)
        current_bundle = replace(bundle, model=current_model)

        unlabeled_mask = np.ones(len(X_pool), dtype=bool)
        labeled_indices: List[int] = []
        labeled_reviewed_labels: List[int] = []

        strategy_start = time.perf_counter()
        for cycle_index in range(1, int(args.cycles) + 1):
            pool_indices = np.where(unlabeled_mask)[0]
            if len(pool_indices) == 0:
                break

            pool_eval = evaluate_model(
                model=current_bundle.model,
                X=X_pool[pool_indices],
                y=y_pool[pool_indices],
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )
            chosen_indices = select_query_indices(
                strategy=strategy,
                X_pool=X_pool,
                probabilities=pool_eval["probabilities"],
                unlabeled_indices=pool_indices,
                query_budget=args.query_budget,
                candidate_multiplier=args.candidate_multiplier,
                rng=strategy_rng,
            )
            if len(chosen_indices) == 0:
                break

            reviewed_labels = simulate_human_oracle(
                y_true=y_pool[chosen_indices],
                labels=bundle.labels,
                accuracy=args.annotator_accuracy,
                rng=strategy_rng,
            )
            chosen_probs = evaluate_model(
                model=current_bundle.model,
                X=X_pool[chosen_indices],
                y=y_pool[chosen_indices],
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )["probabilities"]
            chosen_confidences = extract_softmax_confidences(chosen_probs)
            predicted_labels = np.argmax(chosen_probs, axis=1).astype(np.int64)

            for local_idx, global_idx in enumerate(chosen_indices.tolist()):
                query_rows.append(
                    {
                        "strategy": strategy,
                        "cycle_index": int(cycle_index),
                        "pool_index": int(global_idx),
                        "participant": str(groups_pool[global_idx]),
                        "y_true": int(y_pool[global_idx]),
                        "y_pred": int(predicted_labels[local_idx]),
                        "reviewed_label": int(reviewed_labels[local_idx]),
                        "confidence": float(chosen_confidences[local_idx]),
                        "was_corrected": int(int(reviewed_labels[local_idx]) != int(predicted_labels[local_idx])),
                    }
                )

            unlabeled_mask[chosen_indices] = False
            labeled_indices.extend(int(v) for v in chosen_indices.tolist())
            labeled_reviewed_labels.extend(int(v) for v in reviewed_labels.tolist())

            X_labeled = X_pool[np.array(labeled_indices, dtype=int)]
            y_labeled = np.asarray(labeled_reviewed_labels, dtype=np.int64)

            if len(X_labeled) < 2:
                continue

            X_adapt_train, y_adapt_train, X_adapt_holdout, y_adapt_holdout = split_adaptation_pool(
                X_labeled,
                y_labeled,
                ratio=args.adaptation_split,
                rng=strategy_rng,
            )
            X_mix, y_mix = build_replay_mix(
                X_clean_train=X_train,
                y_clean_train=y_train,
                X_drift_adapt=X_adapt_train,
                y_drift_adapt=y_adapt_train,
                replay_ratio=args.clean_replay_ratio,
                rng=strategy_rng,
            )

            before_clean = evaluate_model(
                model=current_bundle.model,
                X=X_test,
                y=y_test,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )
            before_drift = evaluate_model(
                model=current_bundle.model,
                X=X_eval_drift,
                y=y_eval_drift,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )

            update_start = time.perf_counter()
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
            current_model = adapted_model
            current_bundle = replace(current_bundle, model=current_model)

            after_clean = evaluate_model(
                model=current_bundle.model,
                X=X_test,
                y=y_test,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )
            after_drift = evaluate_model(
                model=current_bundle.model,
                X=X_eval_drift,
                y=y_eval_drift,
                device=current_bundle.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scaler=current_bundle.scaler,
                labels=current_bundle.labels,
                class_names=current_bundle.class_names,
            )
            profile = profile_inference_if_available(
                current_bundle,
                X_eval_drift,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            cycle_rows.append(
                {
                    "strategy": strategy,
                    "cycle_index": int(cycle_index),
                    "labels_added_this_cycle": int(len(chosen_indices)),
                    "cumulative_labels": int(len(labeled_indices)),
                    "clean_macro_f1_before": float(before_clean["macro_f1"]),
                    "clean_macro_f1_after": float(after_clean["macro_f1"]),
                    "drifted_macro_f1_before": float(before_drift["macro_f1"]),
                    "drifted_macro_f1_after": float(after_drift["macro_f1"]),
                    "clean_accuracy_after": float(after_clean["accuracy"]),
                    "drifted_accuracy_after": float(after_drift["accuracy"]),
                    "holdout_best_val_macro_f1": float(adaptation_training["best_val_macro_f1"]),
                    "update_time_sec": float(update_time_sec),
                    "throughput_after": profile.get("throughput_samples_per_sec"),
                    "latency_p50_after": profile.get("latency_per_sample_ms_p50"),
                }
            )

        strategy_total_time = time.perf_counter() - strategy_start
        final_clean = evaluate_model(
            model=current_bundle.model,
            X=X_test,
            y=y_test,
            device=current_bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=current_bundle.scaler,
            labels=current_bundle.labels,
            class_names=current_bundle.class_names,
        )
        final_drift = evaluate_model(
            model=current_bundle.model,
            X=X_eval_drift,
            y=y_eval_drift,
            device=current_bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=current_bundle.scaler,
            labels=current_bundle.labels,
            class_names=current_bundle.class_names,
        )

        summary_rows.append(
            {
                "strategy": strategy,
                "cycles_completed": int(sum(1 for row in cycle_rows if row["strategy"] == strategy)),
                "total_labels_reviewed": int(sum(1 for row in query_rows if row["strategy"] == strategy)),
                "clean_macro_f1_final": float(final_clean["macro_f1"]),
                "drifted_macro_f1_final": float(final_drift["macro_f1"]),
                "clean_macro_f1_gain": float(final_clean["macro_f1"] - base_clean["macro_f1"]),
                "drifted_macro_f1_gain": float(final_drift["macro_f1"] - base_drift["macro_f1"]),
                "total_strategy_time_sec": float(strategy_total_time),
            }
        )
        efficiency_rows.append(
            {
                "strategy": strategy,
                "total_labels_reviewed": int(sum(1 for row in query_rows if row["strategy"] == strategy)),
                "cycles_completed": int(sum(1 for row in cycle_rows if row["strategy"] == strategy)),
                "mean_update_time_sec": float(np.mean([row["update_time_sec"] for row in cycle_rows if row["strategy"] == strategy]))
                if any(row["strategy"] == strategy for row in cycle_rows)
                else 0.0,
                "drifted_macro_f1_final": float(final_drift["macro_f1"]),
            }
        )

    cycle_df = pd.DataFrame(cycle_rows)
    query_df = pd.DataFrame(query_rows)
    summary_df = pd.DataFrame(summary_rows)
    efficiency_df = pd.DataFrame(efficiency_rows)

    save_csv(output_dir / "cycle_metrics.csv", cycle_df)
    save_csv(output_dir / "query_log.csv", query_df)
    save_csv(output_dir / "labeling_efficiency.csv", efficiency_df)
    save_csv(output_dir / "strategy_summary.csv", summary_df)

    summary_payload = {
        "source_model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "config": vars(args),
        "baseline_clean": {
            "macro_f1": float(base_clean["macro_f1"]),
            "accuracy": float(base_clean["accuracy"]),
        },
        "baseline_drifted": {
            "macro_f1": float(base_drift["macro_f1"]),
            "accuracy": float(base_drift["accuracy"]),
        },
        "strategies": summary_rows,
    }
    save_json(output_dir / "active_learning_summary.json", summary_payload)
    plot_active_learning_curve(cycle_df, output_dir / "active_learning_curve.png")

    best_strategy = None
    if not summary_df.empty:
        best_row = summary_df.sort_values("drifted_macro_f1_final", ascending=False).iloc[0]
        best_strategy = str(best_row["strategy"])

    (output_dir / "acceptance_notes.txt").write_text(
        "\n".join(
            [
                "Phase 4 acceptance checklist",
                f"- Strategies evaluated: {', '.join(strategies)}",
                f"- Query log rows: {len(query_df)}",
                f"- Cycle metric rows: {len(cycle_df)}",
                f"- Best strategy on drifted data: {best_strategy or 'N/A'}",
                f"- Strategy summary: {output_dir / 'strategy_summary.csv'}",
                f"- Active learning curve: {output_dir / 'active_learning_curve.png'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "strategies": strategies,
                "best_strategy": best_strategy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

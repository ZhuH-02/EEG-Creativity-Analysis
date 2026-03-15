from __future__ import annotations

import argparse
import copy
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    set_seed,
    summarize_feature_reference,
)
from report_plots import plot_adaptation_comparison, plot_efficiency_comparison, save_figure
from train_pipeline import compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a drift simulation and warm-start adaptation experiment.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/final_eval")
    parser.add_argument("--run_name", type=str, default="adaptation_eval")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift_noise", type=float, default=0.05)
    parser.add_argument("--drift_scale", type=float, default=1.2)
    parser.add_argument("--drift_mean_shift", type=float, default=0.25)
    parser.add_argument("--class_prior_strength", type=float, default=0.0)
    parser.add_argument("--adaptation_epochs", type=int, default=12)
    parser.add_argument("--adaptation_patience", type=int, default=4)
    parser.add_argument("--adaptation_lr", type=float, default=5e-4)
    parser.add_argument("--clean_replay_ratio", type=float, default=0.5)
    parser.add_argument("--adaptation_split", type=float, default=0.8)
    return parser.parse_args()


def simulate_feature_drift(
    X: np.ndarray,
    reference_stats: pd.DataFrame,
    noise_scale: float,
    scale_factor: float,
    mean_shift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    stats = reference_stats.set_index("feature")
    std = np.maximum(stats["std"].to_numpy(dtype=np.float64), 1e-8)
    shifted = np.asarray(X, dtype=np.float64) * float(scale_factor)
    shifted = shifted + (float(mean_shift) * std)
    shifted = shifted + rng.normal(0.0, float(noise_scale), size=shifted.shape) * std
    return shifted


def apply_class_prior_shift(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float(strength) <= 0.0:
        return X, y, groups
    labels, counts = np.unique(y, return_counts=True)
    major_label = int(labels[np.argmax(counts)])
    keep_mask = np.zeros(len(y), dtype=bool)
    for label in labels:
        label_idx = np.where(y == int(label))[0]
        if int(label) == major_label:
            keep_count = len(label_idx)
        else:
            keep_count = max(5, int(round(len(label_idx) * max(0.1, 1.0 - float(strength)))))
        chosen = rng.choice(label_idx, size=min(keep_count, len(label_idx)), replace=False)
        keep_mask[chosen] = True
    return X[keep_mask], y[keep_mask], groups[keep_mask]


def split_adaptation_pool(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(X), dtype=int)
    rng.shuffle(indices)
    split_point = max(1, min(len(indices) - 1, int(round(len(indices) * float(ratio)))))
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def build_replay_mix(
    X_clean_train: np.ndarray,
    y_clean_train: np.ndarray,
    X_drift_adapt: np.ndarray,
    y_drift_adapt: np.ndarray,
    replay_ratio: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    replay_count = int(round(len(X_drift_adapt) * max(0.0, float(replay_ratio))))
    replay_count = min(max(replay_count, 0), len(X_clean_train))
    if replay_count > 0:
        replay_idx = rng.choice(np.arange(len(X_clean_train), dtype=int), size=replay_count, replace=False)
        X_mix = np.vstack([X_drift_adapt, X_clean_train[replay_idx]])
        y_mix = np.concatenate([y_drift_adapt, y_clean_train[replay_idx]])
    else:
        X_mix = np.asarray(X_drift_adapt, dtype=np.float64)
        y_mix = np.asarray(y_drift_adapt, dtype=np.int64)
    return X_mix, y_mix


def fine_tune_model(
    bundle: ArtifactBundle,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    patience: int,
    learning_rate: float,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    model = copy.deepcopy(bundle.model).to(bundle.device)
    X_train_scaled = bundle.scaler.transform(X_train).astype(np.float32)
    X_val_scaled = bundle.scaler.transform(X_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_scaled),
        torch.from_numpy(np.asarray(y_val, dtype=np.int64)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(int(batch_size), max(1, len(train_dataset))),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(int(batch_size), max(1, len(val_dataset))),
        shuffle=False,
        num_workers=int(num_workers),
    )

    class_weights = compute_class_weights(y_train=np.asarray(y_train), num_classes=max(bundle.labels) + 1, device=bundle.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(bundle.config.get("weight_decay", 1e-4)))

    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -np.inf
    no_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(bundle.device)
            y_batch = y_batch.to(bundle.device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            batch_size_now = int(y_batch.size(0))
            running_loss += float(loss.item()) * batch_size_now
            seen += batch_size_now

        model.eval()
        val_probs: List[np.ndarray] = []
        val_true: List[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(bundle.device))
                val_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                val_true.append(y_batch.numpy())
        probs = np.vstack(val_probs) if val_probs else np.zeros((0, 0), dtype=np.float64)
        y_true = np.concatenate(val_true) if val_true else np.zeros((0,), dtype=np.int64)
        metrics = compute_common_metrics(y_true=y_true, y_pred=np.argmax(probs, axis=1), probabilities=probs, labels=bundle.labels, class_names=bundle.class_names)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(running_loss / max(seen, 1)),
                "val_macro_f1": float(metrics["macro_f1"]),
                "val_accuracy": float(metrics["accuracy"]),
            }
        )
        if float(metrics["macro_f1"]) > best_macro_f1 + 1e-8:
            best_macro_f1 = float(metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_val_macro_f1": best_macro_f1}


def evaluate_before_after(
    bundle: ArtifactBundle,
    adapted_model: torch.nn.Module,
    X_clean_test: np.ndarray,
    y_clean_test: np.ndarray,
    X_drift_test: np.ndarray,
    y_drift_test: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    before_clean = evaluate_model(
        model=bundle.model,
        X=X_clean_test,
        y=y_clean_test,
        device=bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )
    before_drift = evaluate_model(
        model=bundle.model,
        X=X_drift_test,
        y=y_drift_test,
        device=bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )
    after_clean = evaluate_model(
        model=adapted_model,
        X=X_clean_test,
        y=y_clean_test,
        device=bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )
    after_drift = evaluate_model(
        model=adapted_model,
        X=X_drift_test,
        y=y_drift_test,
        device=bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )

    comparison_df = pd.DataFrame(
        [
            {"dataset": "clean_test", "stage": "before", "accuracy": before_clean["accuracy"], "macro_f1": before_clean["macro_f1"], "weighted_f1": before_clean["weighted_f1"]},
            {"dataset": "clean_test", "stage": "after", "accuracy": after_clean["accuracy"], "macro_f1": after_clean["macro_f1"], "weighted_f1": after_clean["weighted_f1"]},
            {"dataset": "drifted_test", "stage": "before", "accuracy": before_drift["accuracy"], "macro_f1": before_drift["macro_f1"], "weighted_f1": before_drift["weighted_f1"]},
            {"dataset": "drifted_test", "stage": "after", "accuracy": after_drift["accuracy"], "macro_f1": after_drift["macro_f1"], "weighted_f1": after_drift["weighted_f1"]},
        ]
    )
    details = {
        "before_clean": before_clean,
        "before_drift": before_drift,
        "after_clean": after_clean,
        "after_drift": after_drift,
    }
    return comparison_df, details


def adaptation_efficiency_to_frame(efficiency_payload: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert adaptation efficiency JSON payload to a plot-ready dataframe."""
    rows: List[Dict[str, Any]] = []
    for condition, metrics in efficiency_payload.items():
        row = {"scenario": str(condition)}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = make_output_dir(args.output_dir, args.run_name)

    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_train, y_train, _ = dataset.subset("train")
    X_val, y_val, groups_val = dataset.subset("val")
    X_test, y_test, groups_test = dataset.subset("test")

    reference_stats = summarize_feature_reference(X_train, bundle.feature_names)
    X_val_drifted = simulate_feature_drift(
        X_val,
        reference_stats=reference_stats,
        noise_scale=args.drift_noise,
        scale_factor=args.drift_scale,
        mean_shift=args.drift_mean_shift,
        rng=rng,
    )
    X_test_drifted = simulate_feature_drift(
        X_test,
        reference_stats=reference_stats,
        noise_scale=args.drift_noise,
        scale_factor=args.drift_scale,
        mean_shift=args.drift_mean_shift,
        rng=rng,
    )
    X_val_drifted, y_val_drifted, groups_val_drifted = apply_class_prior_shift(
        X_val_drifted,
        y_val,
        groups_val,
        strength=args.class_prior_strength,
        rng=rng,
    )
    X_test_drifted, y_test_drifted, groups_test_drifted = apply_class_prior_shift(
        X_test_drifted,
        y_test,
        groups_test,
        strength=args.class_prior_strength,
        rng=rng,
    )

    X_adapt_train, y_adapt_train, X_adapt_holdout, y_adapt_holdout = split_adaptation_pool(
        X_val_drifted,
        y_val_drifted,
        ratio=args.adaptation_split,
        rng=rng,
    )
    X_mix, y_mix = build_replay_mix(
        X_clean_train=X_train,
        y_clean_train=y_train,
        X_drift_adapt=X_adapt_train,
        y_drift_adapt=y_adapt_train,
        replay_ratio=args.clean_replay_ratio,
        rng=rng,
    )

    adapted_model, adaptation_training = fine_tune_model(
        bundle=bundle,
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

    comparison_df, details = evaluate_before_after(
        bundle=bundle,
        adapted_model=adapted_model,
        X_clean_test=X_test,
        y_clean_test=y_test,
        X_drift_test=X_test_drifted,
        y_drift_test=y_test_drifted,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    adapted_bundle = replace(bundle, model=adapted_model)
    adaptation_efficiency = {
        "before_clean": profile_inference_if_available(bundle, X_test, batch_size=args.batch_size, num_workers=args.num_workers),
        "after_clean": profile_inference_if_available(adapted_bundle, X_test, batch_size=args.batch_size, num_workers=args.num_workers),
        "before_drift": profile_inference_if_available(bundle, X_test_drifted, batch_size=args.batch_size, num_workers=args.num_workers),
        "after_drift": profile_inference_if_available(adapted_bundle, X_test_drifted, batch_size=args.batch_size, num_workers=args.num_workers),
    }

    before_drift = details["before_drift"]
    after_drift = details["after_drift"]
    resolved_mask = (before_drift["y_true"] != before_drift["y_pred"]) & (after_drift["y_true"] == after_drift["y_pred"])
    resolved_examples = pd.DataFrame(
        {
            "row_index": np.arange(len(y_test_drifted), dtype=int),
            "participant": groups_test_drifted.astype(str),
            "y_true": before_drift["y_true"],
            "y_pred_before": before_drift["y_pred"],
            "y_pred_after": after_drift["y_pred"],
            "confidence_before": before_drift["confidences"],
            "confidence_after": after_drift["confidences"],
        }
    )
    resolved_examples = resolved_examples[resolved_mask].sort_values(["confidence_after", "confidence_before"], ascending=[False, False])

    adaptation_metrics = {
        "drift_simulation": {
            "noise_scale": float(args.drift_noise),
            "scale_factor": float(args.drift_scale),
            "mean_shift": float(args.drift_mean_shift),
            "class_prior_strength": float(args.class_prior_strength),
        },
        "adaptation": {
            "strategy": "warm_start_fine_tune_with_clean_replay",
            "epochs": int(args.adaptation_epochs),
            "patience": int(args.adaptation_patience),
            "learning_rate": float(args.adaptation_lr),
            "clean_replay_ratio": float(args.clean_replay_ratio),
            "adaptation_split": float(args.adaptation_split),
            "training_summary": adaptation_training,
        },
        "results": comparison_df.to_dict(orient="records"),
    }

    save_csv(output_dir / "adaptation_before_after.csv", comparison_df)
    save_json(output_dir / "adaptation_metrics.json", adaptation_metrics)
    save_json(output_dir / "adaptation_efficiency.json", adaptation_efficiency)
    save_csv(output_dir / "resolved_failure_examples.csv", resolved_examples.head(250))

    saved_comparison_df = pd.read_csv(output_dir / "adaptation_before_after.csv")
    adaptation_fig = plot_adaptation_comparison(saved_comparison_df, title="Adaptation Before vs After")
    save_figure(adaptation_fig, output_dir / "adaptation_before_after.png")

    adaptation_efficiency_df = adaptation_efficiency_to_frame(adaptation_efficiency)
    efficiency_fig = plot_efficiency_comparison(
        adaptation_efficiency_df,
        title="Efficiency Comparison Across Adaptation Conditions",
    )
    save_figure(efficiency_fig, output_dir / "efficiency_comparison.png")

    torch.save(
        {
            "model_name": bundle.model_name,
            "model_state_dict": adapted_model.state_dict(),
            "scaler_mean": bundle.scaler.mean.tolist() if bundle.scaler.mean is not None else None,
            "scaler_std": bundle.scaler.std.tolist() if bundle.scaler.std is not None else None,
            "feature_names": bundle.feature_names,
            "labels": bundle.labels,
            "class_names": bundle.class_names,
            "source_model_dir": str(bundle.model_dir),
        },
        output_dir / "adapted_model.pt",
    )

    print(textwrap.dedent(
        f"""
        Adaptation evaluation complete.
        Output directory: {output_dir}
        Drifted macro-F1 before: {details['before_drift']['macro_f1']:.4f}
        Drifted macro-F1 after:  {details['after_drift']['macro_f1']:.4f}
        Resolved failures: {len(resolved_examples)}
        """
    ).strip())


if __name__ == "__main__":
    main()

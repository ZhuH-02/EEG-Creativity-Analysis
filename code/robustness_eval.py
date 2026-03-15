from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from analysis_utils import (
    compute_common_metrics,
    confidence_threshold_predictions,
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
from report_plots import (
    plot_confidence_histogram,
    plot_efficiency_comparison,
    plot_reliability_diagram,
    plot_robustness_curve,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness, calibration, and adversarial evaluation on a trained EEG model.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/final_eval")
    parser.add_argument("--run_name", type=str, default="robustness_eval")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_levels", type=str, default="0.0,0.01,0.05,0.1")
    parser.add_argument("--mask_rates", type=str, default="0.0,0.05,0.15,0.3")
    parser.add_argument("--fgsm_epsilons", type=str, default="0.0,0.01,0.03,0.05,0.1")
    parser.add_argument("--confidence_threshold", type=float, default=0.6)
    parser.add_argument("--calibration_bins", type=int, default=15)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def predict_logits(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(np.asarray(X_scaled, dtype=np.float32)))
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(1, len(dataset))),
        shuffle=False,
        num_workers=int(num_workers),
    )
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for (X_batch,) in loader:
            logits = model(X_batch.to(device))
            chunks.append(logits.cpu().numpy())
    return np.vstack(chunks).astype(np.float64) if chunks else np.zeros((0, 0), dtype=np.float64)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits, dtype=np.float64)
    shifted = logits_arr - np.max(logits_arr, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(np.sum(exp_values, axis=1, keepdims=True), 1e-12)


def negative_log_likelihood(logits: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.clip(softmax_numpy(logits), 1e-12, 1.0)
    losses = -np.log(probs[np.arange(len(y_true)), np.asarray(y_true, dtype=np.int64)])
    return float(np.mean(losses)) if len(losses) else 0.0


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, device: torch.device) -> float:
    if len(logits) == 0:
        return 1.0
    logits_tensor = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(y_true, dtype=torch.long, device=device)
    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().cpu().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, pd.DataFrame]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int64)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correctness = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: List[Dict[str, float]] = []
    ece = 0.0
    total = max(len(labels), 1)
    for idx in range(int(n_bins)):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == int(n_bins) - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            rows.append(
                {
                    "bin_index": idx,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "bin_confidence": float((left + right) / 2.0),
                    "bin_accuracy": 0.0,
                    "bin_fraction": 0.0,
                }
            )
            continue
        bin_confidence = float(np.mean(confidences[mask]))
        bin_accuracy = float(np.mean(correctness[mask]))
        bin_fraction = float(np.mean(mask))
        ece += abs(bin_accuracy - bin_confidence) * (np.sum(mask) / total)
        rows.append(
            {
                "bin_index": idx,
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_confidence": bin_confidence,
                "bin_accuracy": bin_accuracy,
                "bin_fraction": bin_fraction,
            }
        )
    return float(ece), pd.DataFrame(rows)


def gaussian_noise_stress(
    X: np.ndarray,
    reference_stats: pd.DataFrame,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    std = np.maximum(reference_stats["std"].to_numpy(dtype=np.float64), 1e-8)
    return np.asarray(X, dtype=np.float64) + rng.normal(0.0, float(noise_level), size=X.shape) * std


def random_feature_masking(X: np.ndarray, mask_rate: float, rng: np.random.Generator) -> np.ndarray:
    perturbed = np.asarray(X, dtype=np.float64).copy()
    mask = rng.random(size=perturbed.shape) < float(mask_rate)
    perturbed[mask] = 0.0
    return perturbed


def evaluate_probabilities(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    return compute_common_metrics(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=predictions,
        probabilities=np.asarray(probabilities, dtype=np.float64),
        labels=labels,
        class_names=class_names,
    )


def abstention_metrics(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    predictions, accepted_mask = confidence_threshold_predictions(probabilities, threshold=threshold)
    abstain_rate = float(1.0 - np.mean(accepted_mask)) if len(accepted_mask) else 0.0
    if not np.any(accepted_mask):
        return {
            "coverage": 0.0,
            "abstention_rate": abstain_rate,
            "accepted_accuracy": 0.0,
            "accepted_macro_f1": 0.0,
            "accepted_weighted_f1": 0.0,
        }

    accepted_result = compute_common_metrics(
        y_true=np.asarray(y_true, dtype=np.int64)[accepted_mask],
        y_pred=predictions[accepted_mask],
        probabilities=np.asarray(probabilities, dtype=np.float64)[accepted_mask],
        labels=labels,
        class_names=class_names,
    )
    return {
        "coverage": float(np.mean(accepted_mask)),
        "abstention_rate": abstain_rate,
        "accepted_accuracy": float(accepted_result["accuracy"]),
        "accepted_macro_f1": float(accepted_result["macro_f1"]),
        "accepted_weighted_f1": float(accepted_result["weighted_f1"]),
    }


def fgsm_attack(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
    epsilon: float,
    batch_size: int,
    num_workers: int,
    clip_min: np.ndarray,
    clip_max: np.ndarray,
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X_scaled, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_true, dtype=np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), max(1, len(dataset))),
        shuffle=False,
        num_workers=int(num_workers),
    )
    criterion = nn.CrossEntropyLoss()
    adv_batches: List[np.ndarray] = []
    clip_min_tensor = torch.tensor(clip_min, dtype=torch.float32, device=device)
    clip_max_tensor = torch.tensor(clip_max, dtype=torch.float32, device=device)

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device).clone().detach().requires_grad_(True)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        model.zero_grad(set_to_none=True)
        loss.backward()
        adv = X_batch + float(epsilon) * X_batch.grad.sign()
        adv = torch.max(torch.min(adv, clip_max_tensor), clip_min_tensor)
        adv_batches.append(adv.detach().cpu().numpy())

    return np.vstack(adv_batches).astype(np.float64) if adv_batches else np.zeros_like(X_scaled, dtype=np.float64)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = make_output_dir(args.output_dir, args.run_name)

    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_train, _, _ = dataset.subset("train")
    X_val, y_val, _ = dataset.subset("val")
    X_test, y_test, _ = dataset.subset("test")

    reference_stats = summarize_feature_reference(X_train, bundle.feature_names)
    rng = np.random.default_rng(args.seed)

    clean_result = evaluate_model(
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

    val_scaled = bundle.scaler.transform(X_val).astype(np.float32)
    test_scaled = bundle.scaler.transform(X_test).astype(np.float32)
    train_scaled = bundle.scaler.transform(X_train).astype(np.float32)

    val_logits = predict_logits(bundle.model, val_scaled, bundle.device, args.batch_size, args.num_workers)
    test_logits = predict_logits(bundle.model, test_scaled, bundle.device, args.batch_size, args.num_workers)
    temperature = fit_temperature(val_logits, y_val, bundle.device)

    raw_test_probs = softmax_numpy(test_logits)
    scaled_test_probs = softmax_numpy(apply_temperature(test_logits, temperature))
    raw_ece, raw_calibration_df = expected_calibration_error(raw_test_probs, y_test, n_bins=args.calibration_bins)
    scaled_ece, scaled_calibration_df = expected_calibration_error(scaled_test_probs, y_test, n_bins=args.calibration_bins)

    temperature_result = evaluate_probabilities(scaled_test_probs, y_test, bundle.labels, bundle.class_names)
    abstention_result = abstention_metrics(
        probabilities=scaled_test_probs,
        y_true=y_test,
        threshold=args.confidence_threshold,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )

    robustness_rows: List[Dict[str, Any]] = [
        {
            "scenario": "clean",
            "severity": 0.0,
            "accuracy": float(clean_result["accuracy"]),
            "macro_f1": float(clean_result["macro_f1"]),
            "weighted_f1": float(clean_result["weighted_f1"]),
            "mean_confidence": float(np.mean(clean_result["confidences"])),
        },
        {
            "scenario": "temperature_scaled",
            "severity": float(temperature),
            "accuracy": float(temperature_result["accuracy"]),
            "macro_f1": float(temperature_result["macro_f1"]),
            "weighted_f1": float(temperature_result["weighted_f1"]),
            "mean_confidence": float(np.mean(temperature_result["confidences"])),
        },
        {
            "scenario": "abstention",
            "severity": float(args.confidence_threshold),
            "accuracy": float(abstention_result["accepted_accuracy"]),
            "macro_f1": float(abstention_result["accepted_macro_f1"]),
            "weighted_f1": float(abstention_result["accepted_weighted_f1"]),
            "mean_confidence": float(np.mean(temperature_result["confidences"])),
            "coverage": float(abstention_result["coverage"]),
            "abstention_rate": float(abstention_result["abstention_rate"]),
        },
    ]

    for noise_level in parse_float_list(args.noise_levels):
        perturbed_X = gaussian_noise_stress(X_test, reference_stats, noise_level, rng=rng)
        metrics = evaluate_model(
            model=bundle.model,
            X=perturbed_X,
            y=y_test,
            device=bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=bundle.scaler,
            labels=bundle.labels,
            class_names=bundle.class_names,
        )
        robustness_rows.append(
            {
                "scenario": "gaussian_noise",
                "severity": float(noise_level),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "mean_confidence": float(np.mean(metrics["confidences"])),
            }
        )

    for mask_rate in parse_float_list(args.mask_rates):
        perturbed_X = random_feature_masking(X_test, mask_rate, rng=rng)
        metrics = evaluate_model(
            model=bundle.model,
            X=perturbed_X,
            y=y_test,
            device=bundle.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            scaler=bundle.scaler,
            labels=bundle.labels,
            class_names=bundle.class_names,
        )
        robustness_rows.append(
            {
                "scenario": "feature_masking",
                "severity": float(mask_rate),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "mean_confidence": float(np.mean(metrics["confidences"])),
            }
        )

    clip_min = train_scaled.min(axis=0)
    clip_max = train_scaled.max(axis=0)
    fgsm_rows: List[Dict[str, Any]] = []
    fgsm_results_by_eps: Dict[float, Dict[str, Any]] = {}
    for epsilon in parse_float_list(args.fgsm_epsilons):
        adv_scaled = fgsm_attack(
            model=bundle.model,
            X_scaled=test_scaled,
            y_true=y_test,
            device=bundle.device,
            epsilon=epsilon,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        adv_logits = predict_logits(bundle.model, adv_scaled, bundle.device, args.batch_size, args.num_workers)
        adv_probs = softmax_numpy(adv_logits)
        adv_metrics = evaluate_probabilities(adv_probs, y_test, bundle.labels, bundle.class_names)
        fgsm_rows.append(
            {
                "epsilon": float(epsilon),
                "accuracy": float(adv_metrics["accuracy"]),
                "macro_f1": float(adv_metrics["macro_f1"]),
                "weighted_f1": float(adv_metrics["weighted_f1"]),
                "mean_confidence": float(np.mean(adv_metrics["confidences"])) if len(adv_metrics["confidences"]) else 0.0,
            }
        )
        fgsm_results_by_eps[float(epsilon)] = {
            "scaled_inputs": adv_scaled,
            "metrics": adv_metrics,
        }

    robustness_df = pd.DataFrame(robustness_rows)
    fgsm_df = pd.DataFrame(fgsm_rows).sort_values("epsilon").reset_index(drop=True)

    calibration_metrics = {
        "temperature": float(temperature),
        "raw_nll": float(negative_log_likelihood(test_logits, y_test)),
        "temperature_scaled_nll": float(negative_log_likelihood(apply_temperature(test_logits, temperature), y_test)),
        "raw_ece": float(raw_ece),
        "temperature_scaled_ece": float(scaled_ece),
        "abstention": abstention_result,
    }

    clean_efficiency = profile_inference_if_available(bundle, X_test, batch_size=args.batch_size, num_workers=args.num_workers)
    noisy_efficiency = profile_inference_if_available(
        bundle,
        gaussian_noise_stress(X_test, reference_stats, max(parse_float_list(args.noise_levels)), rng=rng),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    max_epsilon = max(float(v) for v in fgsm_results_by_eps)
    adv_efficiency = profile_inference_if_available(
        bundle,
        fgsm_results_by_eps[max_epsilon]["scaled_inputs"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        already_scaled=True,
    )
    efficiency_df = pd.DataFrame(
        [
            {"scenario": "clean", **clean_efficiency},
            {"scenario": "gaussian_noise_max", **noisy_efficiency},
            {"scenario": f"fgsm_eps_{max_epsilon:.3f}", **adv_efficiency},
        ]
    )

    reliability_fig = plot_reliability_diagram(scaled_calibration_df, title="Reliability Diagram (Temperature Scaled)")
    confidence_clean_fig = plot_confidence_histogram(
        confidences=temperature_result["confidences"],
        correct_mask=temperature_result["y_true"] == temperature_result["y_pred"],
        title="Confidence Histogram (Clean, Temperature Scaled)",
    )
    confidence_adv_fig = plot_confidence_histogram(
        confidences=fgsm_results_by_eps[max_epsilon]["metrics"]["confidences"],
        correct_mask=fgsm_results_by_eps[max_epsilon]["metrics"]["y_true"] == fgsm_results_by_eps[max_epsilon]["metrics"]["y_pred"],
        title=f"Confidence Histogram (FGSM epsilon={max_epsilon:.3f})",
    )
    robustness_curve_fig = plot_robustness_curve(
        df=fgsm_df,
        x_col="epsilon",
        y_col="macro_f1",
        title="FGSM Robustness Curve",
    )
    efficiency_comparison_fig = plot_efficiency_comparison(
        efficiency_df,
        title="Efficiency Comparison Across Robustness Conditions",
    )

    save_csv(output_dir / "robustness_metrics.csv", robustness_df)
    save_csv(output_dir / "fgsm_metrics.csv", fgsm_df)
    save_json(output_dir / "calibration_metrics.json", calibration_metrics)
    save_csv(output_dir / "clean_vs_perturbed_efficiency.csv", efficiency_df)
    save_csv(output_dir / "reliability_bins.csv", scaled_calibration_df)
    save_figure(robustness_curve_fig, output_dir / "robustness_curve.png")
    save_figure(reliability_fig, output_dir / "reliability_diagram.png")
    save_figure(confidence_clean_fig, output_dir / "confidence_histogram_clean.png")
    save_figure(confidence_adv_fig, output_dir / "confidence_histogram_adv.png")
    save_figure(efficiency_comparison_fig, output_dir / "efficiency_comparison.png")
    (output_dir / "clean_temperature_scaled_report.txt").write_text(temperature_result["classification_report"], encoding="utf-8")
    (output_dir / "calibration_summary.json").write_text(json.dumps({"raw_bins": raw_calibration_df.to_dict(orient="records")[:3]}, indent=2), encoding="utf-8")

    print(textwrap.dedent(
        f"""
        Robustness evaluation complete.
        Output directory: {output_dir}
        Clean macro-F1: {clean_result['macro_f1']:.4f}
        Temperature-scaled ECE: {scaled_ece:.4f}
        FGSM worst macro-F1: {fgsm_df['macro_f1'].min():.4f}
        """
    ).strip())


if __name__ == "__main__":
    main()

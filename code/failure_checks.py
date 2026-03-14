from __future__ import annotations

import argparse
import concurrent.futures
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis_utils import (
    ArtifactBundle,
    compute_common_metrics,
    confidence_threshold_predictions,
    ensure_non_empty,
    evaluate_model,
    load_dataset_for_run,
    load_trained_artifacts,
    make_output_dir,
    save_csv,
    save_json,
    set_seed,
    summarize_feature_reference,
)


FAILURE_CATALOG: List[Dict[str, str]] = [
    {
        "name": "covariate_shift",
        "stage": "input",
        "description": "Feature distribution changes relative to training windows can degrade learned decision boundaries.",
        "mitigation": "Monitor PSI and Wasserstein distance, retrain or recalibrate when drift persists.",
    },
    {
        "name": "label_shift",
        "stage": "monitoring",
        "description": "Class prior changes can bias weighted metrics and confidence behavior.",
        "mitigation": "Track prediction class mix and recalibrate thresholds when priors move.",
    },
    {
        "name": "concept_drift",
        "stage": "model",
        "description": "The relationship between EEG features and task phases may change over time or by cohort.",
        "mitigation": "Run periodic adaptation experiments and evaluate on fresh held-out participants.",
    },
    {
        "name": "missing_critical_features",
        "stage": "input",
        "description": "Dropped or renamed features break checkpoint compatibility.",
        "mitigation": "Validate feature schema before inference and reject incompatible payloads.",
    },
    {
        "name": "out_of_range_feature_values",
        "stage": "input",
        "description": "Extreme values indicate sensor issues, preprocessing bugs, or unseen operating conditions.",
        "mitigation": "Compare against training reference ranges and flag anomalous windows.",
    },
    {
        "name": "sensor_channel_dropout",
        "stage": "signal",
        "description": "Missing channels alter the feature representation and can create brittle predictions.",
        "mitigation": "Stress-test with feature dropout and guard upstream sensor health.",
    },
    {
        "name": "malformed_json_or_wrong_input_shape",
        "stage": "runtime",
        "description": "Bad payload shape or schema should fail fast instead of producing silent garbage outputs.",
        "mitigation": "Reject empty inputs, wrong dimensionality, and malformed feature tables.",
    },
    {
        "name": "nan_inf_values",
        "stage": "runtime",
        "description": "NaN and Inf values can poison scaling, logits, and metrics.",
        "mitigation": "Block inference when non-finite values are detected.",
    },
    {
        "name": "overconfident_wrong_predictions",
        "stage": "post_prediction",
        "description": "High-confidence errors are risky in downstream decision support.",
        "mitigation": "Track confidence calibration and support abstention thresholds.",
    },
    {
        "name": "class_rarity_imbalance_degradation",
        "stage": "evaluation",
        "description": "Rare classes can collapse first under shift and imbalance.",
        "mitigation": "Report macro-F1, per-class metrics, and stress-test skewed class mixes.",
    },
    {
        "name": "missing_checkpoint_or_schema_mismatch",
        "stage": "startup",
        "description": "Broken run folders or incompatible checkpoints should stop the evaluation cleanly.",
        "mitigation": "Validate required files before loading and surface clear errors.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anticipated failure checks and stress tests for a trained EEG model.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs/final_eval")
    parser.add_argument("--run_name", type=str, default="failure_checks")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_levels", type=str, default="0.01,0.05,0.1")
    parser.add_argument("--mask_rates", type=str, default="0.05,0.15,0.3")
    parser.add_argument("--dropout_rates", type=str, default="0.1,0.25,0.4")
    parser.add_argument("--amplitude_scales", type=str, default="0.7,1.3,1.6")
    parser.add_argument("--confidence_threshold", type=float, default=0.55)
    parser.add_argument("--high_confidence_error_threshold", type=float, default=0.85)
    parser.add_argument("--missing_threshold", type=float, default=0.2)
    parser.add_argument("--range_std_threshold", type=float, default=4.0)
    parser.add_argument("--timeout_sec", type=float, default=30.0)
    parser.add_argument("--max_batch_size", type=int, default=8192)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def render_failure_catalog_markdown(catalog: Sequence[Dict[str, str]]) -> str:
    lines = ["# Failure Catalog", ""]
    for item in catalog:
        lines.append(f"## {item['name']}")
        lines.append(f"- Stage: {item['stage']}")
        lines.append(f"- Description: {item['description']}")
        lines.append(f"- Mitigation: {item['mitigation']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def validate_feature_schema(input_df: pd.DataFrame, expected_features: Sequence[str]) -> Dict[str, Any]:
    expected = list(expected_features)
    missing = [feature for feature in expected if feature not in input_df.columns]
    extra = [feature for feature in input_df.columns if feature not in expected]
    reordered = [feature for feature in expected if feature in input_df.columns]
    return {
        "valid": not missing,
        "missing_features": missing,
        "extra_features": extra,
        "reordered_columns": reordered,
    }


def check_missingness(input_df: pd.DataFrame, threshold: float = 0.2) -> Dict[str, Any]:
    missing_rates = input_df.isna().mean(axis=0).sort_values(ascending=False)
    flagged = missing_rates[missing_rates > float(threshold)]
    return {
        "threshold": float(threshold),
        "overall_missing_rate": float(input_df.isna().mean().mean()),
        "flagged_features": flagged.index.tolist(),
        "feature_missing_rates": {str(index): float(value) for index, value in missing_rates.items()},
    }


def check_feature_ranges(
    input_df: pd.DataFrame,
    reference_stats: pd.DataFrame,
    std_threshold: float = 4.0,
) -> Dict[str, Any]:
    stats = reference_stats.set_index("feature")
    flags: Dict[str, Dict[str, float]] = {}
    for feature in input_df.columns:
        if feature not in stats.index:
            continue
        values = input_df[feature].astype(float)
        ref_mean = float(stats.loc[feature, "mean"])
        ref_std = max(float(stats.loc[feature, "std"]), 1e-8)
        z_scores = np.abs((values - ref_mean) / ref_std)
        outlier_rate = float(np.mean(z_scores > float(std_threshold)))
        hard_min = float(stats.loc[feature, "q01"])
        hard_max = float(stats.loc[feature, "q99"])
        tail_rate = float(np.mean((values < hard_min) | (values > hard_max)))
        if outlier_rate > 0.0 or tail_rate > 0.0:
            flags[feature] = {
                "outlier_rate_std": outlier_rate,
                "tail_rate_q01_q99": tail_rate,
            }
    return {
        "std_threshold": float(std_threshold),
        "flagged_features": flags,
        "n_flagged_features": int(len(flags)),
    }


def check_window_length(actual_length: int, expected_length: int) -> Dict[str, Any]:
    return {
        "expected_window_length": int(expected_length),
        "actual_window_length": int(actual_length),
        "valid": int(actual_length) == int(expected_length),
    }


def check_no_nan_inf(input_df: pd.DataFrame) -> Dict[str, Any]:
    values = input_df.to_numpy(dtype=np.float64)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    return {
        "valid": nan_count == 0 and inf_count == 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def reject_wrong_dimensionality(X: np.ndarray, expected_dim: int) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, received shape {tuple(X.shape)}")
    if X.shape[1] != int(expected_dim):
        raise ValueError(f"Expected {expected_dim} features, received {X.shape[1]}")


def reject_empty_input(X: np.ndarray) -> None:
    if X.size == 0 or len(X) == 0:
        raise ValueError("Received empty input for inference.")


def reject_nan_inf(X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        raise ValueError("Received NaN or Inf values in input features.")


def batch_size_guard(batch_size: int, max_batch_size: int) -> None:
    if int(batch_size) <= 0:
        raise ValueError("Batch size must be > 0.")
    if int(batch_size) > int(max_batch_size):
        raise ValueError(f"Batch size {batch_size} exceeds configured guard {max_batch_size}.")


def run_with_timeout(func: Callable[..., Any], timeout_sec: float, *args: Any, **kwargs: Any) -> Any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=float(timeout_sec))


def compute_anomaly_scores(X: np.ndarray, reference_stats: pd.DataFrame) -> np.ndarray:
    stats = reference_stats.set_index("feature")
    means = stats["mean"].to_numpy(dtype=np.float64)
    stds = np.maximum(stats["std"].to_numpy(dtype=np.float64), 1e-8)
    z = np.abs((np.asarray(X, dtype=np.float64) - means) / stds)
    return np.max(z, axis=1)


def low_confidence_flags(confidences: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(confidences, dtype=np.float64) < float(threshold)


def suspicious_high_confidence_on_anomalies(
    anomaly_scores: np.ndarray,
    confidences: np.ndarray,
    anomaly_threshold: float = 4.0,
    confidence_threshold: float = 0.9,
) -> np.ndarray:
    return (np.asarray(anomaly_scores) >= float(anomaly_threshold)) & (np.asarray(confidences) >= float(confidence_threshold))


def gaussian_noise_stress(
    X: np.ndarray,
    reference_stats: pd.DataFrame,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    std = np.maximum(reference_stats["std"].to_numpy(dtype=np.float64), 1e-8)
    noise = rng.normal(loc=0.0, scale=float(noise_scale), size=X.shape) * std
    return np.asarray(X, dtype=np.float64) + noise


def random_feature_masking(X: np.ndarray, mask_rate: float, rng: np.random.Generator) -> np.ndarray:
    masked = np.asarray(X, dtype=np.float64).copy()
    mask = rng.random(masked.shape) < float(mask_rate)
    masked[mask] = 0.0
    return masked


def channel_dropout_simulation(
    X: np.ndarray,
    feature_names: Sequence[str],
    dropout_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    # Current features are channel-averaged, so this approximates upstream sensor loss in feature space.
    dropped = np.asarray(X, dtype=np.float64).copy()
    feature_groups: List[List[int]] = []
    time_idx = [idx for idx, name in enumerate(feature_names) if not str(name).endswith("_power")]
    freq_idx = [idx for idx, name in enumerate(feature_names) if str(name).endswith("_power")]
    if time_idx:
        feature_groups.append(time_idx)
    if freq_idx:
        feature_groups.append(freq_idx)
    if not feature_groups:
        return dropped
    for row_idx in range(dropped.shape[0]):
        if rng.random() < float(dropout_rate):
            group = feature_groups[int(rng.integers(0, len(feature_groups)))]
            dropped[row_idx, group] = 0.0
    return dropped


def amplitude_scaling(X: np.ndarray, scale_factor: float) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) * float(scale_factor)


def class_rarity_scenario(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels, counts = np.unique(y, return_counts=True)
    rare_label = int(labels[np.argmin(counts)])
    keep_mask = np.ones(len(y), dtype=bool)
    rare_idx = np.where(y == rare_label)[0]
    if len(rare_idx) > 10:
        keep_count = max(5, int(round(len(rare_idx) * 0.15)))
        chosen = rng.choice(rare_idx, size=keep_count, replace=False)
        keep_mask[rare_idx] = False
        keep_mask[chosen] = True
    return X[keep_mask], y[keep_mask], groups[keep_mask]


def participant_ood_rows(
    bundle: ArtifactBundle,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for participant in sorted(np.unique(groups).tolist()):
        idx = np.where(groups == participant)[0]
        if len(idx) == 0:
            continue
        metrics = evaluate_model(
            model=bundle.model,
            X=X[idx],
            y=y[idx],
            device=bundle.device,
            batch_size=batch_size,
            num_workers=num_workers,
            scaler=bundle.scaler,
            labels=bundle.labels,
            class_names=bundle.class_names,
        )
        rows.append(
            {
                "scenario": "held_out_participant",
                "severity": str(participant),
                "n_samples": int(len(idx)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "mean_confidence": float(np.mean(metrics["confidences"])) if len(metrics["confidences"]) else 0.0,
            }
        )
    return rows


def build_metric_row(
    scenario: str,
    severity: str,
    eval_result: Dict[str, Any],
    y_true: np.ndarray,
    confidence_threshold: float,
) -> Dict[str, Any]:
    _, accepted_mask = confidence_threshold_predictions(eval_result["probabilities"], threshold=confidence_threshold)
    return {
        "scenario": scenario,
        "severity": severity,
        "n_samples": int(len(y_true)),
        "accuracy": float(eval_result["accuracy"]),
        "macro_f1": float(eval_result["macro_f1"]),
        "weighted_f1": float(eval_result["weighted_f1"]),
        "mean_confidence": float(np.mean(eval_result["confidences"])) if len(eval_result["confidences"]) else 0.0,
        "abstention_rate": float(1.0 - np.mean(accepted_mask)) if len(accepted_mask) else 0.0,
    }


def evaluate_stress_case(
    bundle: ArtifactBundle,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    return evaluate_model(
        model=bundle.model,
        X=X,
        y=y,
        device=bundle.device,
        batch_size=batch_size,
        num_workers=num_workers,
        scaler=bundle.scaler,
        labels=bundle.labels,
        class_names=bundle.class_names,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = make_output_dir(args.output_dir, args.run_name)
    bundle = load_trained_artifacts(args.model_dir, device=args.device)
    dataset = load_dataset_for_run(bundle)
    X_eval, y_eval, groups_eval = dataset.subset(args.split)
    X_train, y_train, _ = dataset.subset("train")

    ensure_non_empty(args.split, X_eval, y_eval)
    batch_size_guard(args.batch_size, args.max_batch_size)
    reject_empty_input(X_eval)
    reject_wrong_dimensionality(X_eval, len(bundle.feature_names))
    reject_nan_inf(X_eval)

    eval_df = pd.DataFrame(X_eval, columns=bundle.feature_names)
    reference_stats = summarize_feature_reference(X_train, bundle.feature_names)

    schema_report = validate_feature_schema(eval_df, bundle.feature_names)
    missingness_report = check_missingness(eval_df, threshold=args.missing_threshold)
    range_report = check_feature_ranges(eval_df, reference_stats, std_threshold=args.range_std_threshold)
    finite_report = check_no_nan_inf(eval_df)
    window_report = check_window_length(actual_length=len(bundle.feature_names), expected_length=len(bundle.feature_names))

    clean_result = run_with_timeout(
        evaluate_stress_case,
        args.timeout_sec,
        bundle,
        X_eval,
        y_eval,
        args.batch_size,
        args.num_workers,
    )

    anomaly_scores = compute_anomaly_scores(X_eval, reference_stats)
    low_conf_mask = low_confidence_flags(clean_result["confidences"], threshold=args.confidence_threshold)
    suspicious_mask = suspicious_high_confidence_on_anomalies(
        anomaly_scores=anomaly_scores,
        confidences=clean_result["confidences"],
        anomaly_threshold=args.range_std_threshold,
        confidence_threshold=max(args.high_confidence_error_threshold, args.confidence_threshold),
    )
    overconfident_error_mask = (
        (clean_result["y_true"] != clean_result["y_pred"])
        & (clean_result["confidences"] >= float(args.high_confidence_error_threshold))
    )

    failure_examples = pd.DataFrame(
        {
            "row_index": np.arange(len(y_eval), dtype=int),
            "participant": groups_eval.astype(str),
            "y_true": clean_result["y_true"],
            "y_pred": clean_result["y_pred"],
            "confidence": clean_result["confidences"],
            "anomaly_score": anomaly_scores,
            "low_confidence_flag": low_conf_mask.astype(int),
            "suspicious_high_confidence_flag": suspicious_mask.astype(int),
            "overconfident_error_flag": overconfident_error_mask.astype(int),
        }
    )
    failure_examples = failure_examples[
        (failure_examples["low_confidence_flag"] == 1)
        | (failure_examples["suspicious_high_confidence_flag"] == 1)
        | (failure_examples["overconfident_error_flag"] == 1)
    ].sort_values(["overconfident_error_flag", "confidence", "anomaly_score"], ascending=[False, False, False])

    rng = np.random.default_rng(args.seed)
    stress_rows: List[Dict[str, Any]] = [
        build_metric_row(
            scenario="clean",
            severity="baseline",
            eval_result=clean_result,
            y_true=y_eval,
            confidence_threshold=args.confidence_threshold,
        )
    ]

    for noise_level in parse_float_list(args.noise_levels):
        stressed_X = gaussian_noise_stress(X_eval, reference_stats, noise_level, rng=rng)
        metrics = evaluate_stress_case(bundle, stressed_X, y_eval, args.batch_size, args.num_workers)
        stress_rows.append(build_metric_row("gaussian_noise", f"{noise_level:.3f}", metrics, y_eval, args.confidence_threshold))

    for mask_rate in parse_float_list(args.mask_rates):
        stressed_X = random_feature_masking(X_eval, mask_rate=mask_rate, rng=rng)
        metrics = evaluate_stress_case(bundle, stressed_X, y_eval, args.batch_size, args.num_workers)
        stress_rows.append(build_metric_row("feature_masking", f"{mask_rate:.3f}", metrics, y_eval, args.confidence_threshold))

    for dropout_rate in parse_float_list(args.dropout_rates):
        stressed_X = channel_dropout_simulation(X_eval, bundle.feature_names, dropout_rate=dropout_rate, rng=rng)
        metrics = evaluate_stress_case(bundle, stressed_X, y_eval, args.batch_size, args.num_workers)
        stress_rows.append(build_metric_row("channel_dropout", f"{dropout_rate:.3f}", metrics, y_eval, args.confidence_threshold))

    for scale in parse_float_list(args.amplitude_scales):
        stressed_X = amplitude_scaling(X_eval, scale_factor=scale)
        metrics = evaluate_stress_case(bundle, stressed_X, y_eval, args.batch_size, args.num_workers)
        stress_rows.append(build_metric_row("amplitude_scaling", f"{scale:.3f}", metrics, y_eval, args.confidence_threshold))

    X_rarity, y_rarity, groups_rarity = class_rarity_scenario(X_eval, y_eval, groups_eval, rng=rng)
    rarity_metrics = evaluate_stress_case(bundle, X_rarity, y_rarity, args.batch_size, args.num_workers)
    stress_rows.append(build_metric_row("class_rarity", "rare_class_downsampled", rarity_metrics, y_rarity, args.confidence_threshold))

    stress_rows.extend(
        participant_ood_rows(
            bundle=bundle,
            X=X_eval,
            y=y_eval,
            groups=groups_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    stress_df = pd.DataFrame(stress_rows).sort_values(["scenario", "severity"]).reset_index(drop=True)
    worst_row = stress_df.sort_values(["macro_f1", "accuracy"], ascending=[True, True]).iloc[0].to_dict()

    failure_summary = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": args.split,
        "schema_report": schema_report,
        "missingness_report": missingness_report,
        "range_report": range_report,
        "finite_report": finite_report,
        "window_report": window_report,
        "post_prediction": {
            "confidence_threshold": float(args.confidence_threshold),
            "low_confidence_rate": float(np.mean(low_conf_mask)) if len(low_conf_mask) else 0.0,
            "suspicious_high_confidence_anomaly_rate": float(np.mean(suspicious_mask)) if len(suspicious_mask) else 0.0,
            "overconfident_error_rate": float(np.mean(overconfident_error_mask)) if len(overconfident_error_mask) else 0.0,
        },
        "clean_metrics": {
            "accuracy": float(clean_result["accuracy"]),
            "macro_f1": float(clean_result["macro_f1"]),
            "weighted_f1": float(clean_result["weighted_f1"]),
        },
        "worst_stress_case": worst_row,
        "n_failure_examples": int(len(failure_examples)),
    }

    save_csv(output_dir / "stress_test_metrics.csv", stress_df)
    save_json(output_dir / "failure_catalog.json", FAILURE_CATALOG)
    save_json(output_dir / "failure_summary.json", failure_summary)
    if not failure_examples.empty:
        save_csv(output_dir / "failure_examples.csv", failure_examples.head(250))
    (output_dir / "failure_catalog.md").write_text(render_failure_catalog_markdown(FAILURE_CATALOG), encoding="utf-8")
    (output_dir / "clean_classification_report.txt").write_text(clean_result["classification_report"], encoding="utf-8")

    print(textwrap.dedent(
        f"""
        Failure checks complete.
        Output directory: {output_dir}
        Clean macro-F1: {clean_result['macro_f1']:.4f}
        Worst stress case: {worst_row['scenario']} ({worst_row['severity']}) -> macro-F1={worst_row['macro_f1']:.4f}
        """
    ).strip())


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app import (
    DATA_DIR as DEFAULT_DATA_DIR,
    JSON_PHASE_TO_CANONICAL as DEFAULT_JSON_PHASE_TO_CANONICAL,
    PARTICIPANTS as DEFAULT_PARTICIPANTS,
    PHASE_CODE_MAP as DEFAULT_PHASE_CODE_MAP,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
    SAMPLING_RATE as DEFAULT_SAMPLING_RATE,
    EEGDataLoader,
    EEGFeatureExtractor,
    FeatureStandardizer,
    build_feature_table,
)
from train_milestone2 import (
    build_feature_names,
    format_classification_report,
    make_model,
    multiclass_metrics,
    parse_hidden_layers,
    predict_proba,
    profile_inference,
    resolve_device,
    split_train_val_test_by_subject,
)


@dataclass
class ArtifactBundle:
    model_dir: Path
    checkpoint_path: Path
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    efficiency: Dict[str, Any]
    checkpoint: Dict[str, Any]
    model_name: str
    device: torch.device
    model: torch.nn.Module
    scaler: FeatureStandardizer
    feature_names: List[str]
    labels: List[int]
    class_names: List[str]


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: List[str]
    split_subjects: Dict[str, List[str]]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def subset(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        split_name = split.lower()
        if split_name == "train":
            idx = self.train_idx
        elif split_name == "val":
            idx = self.val_idx
        elif split_name == "test":
            idx = self.test_idx
        else:
            raise ValueError(f"Unknown split: {split}")
        return self.X[idx], self.y[idx], self.groups[idx]


def set_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_output_dir(base_dir: str | Path, run_name: Optional[str] = None) -> Path:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = None
    if run_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name.strip()).strip("-") or None
    folder_name = timestamp if not safe_name else f"{timestamp}_{safe_name}"
    output_dir = base_path / folder_name
    suffix = 1
    while output_dir.exists():
        output_dir = base_path / f"{folder_name}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def load_json_file(path: str | Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return dict(default or {})
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"value": payload}


def save_json(path: str | Path, obj: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def save_csv(path: str | Path, df: pd.DataFrame) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def save_plot(path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close()


def _coerce_labels(labels: Optional[Sequence[int]], probabilities: np.ndarray) -> List[int]:
    if labels is not None:
        return [int(label) for label in labels]
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        return []
    return list(range(int(probabilities.shape[1])))


def extract_softmax_confidences(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.max(probabilities, axis=1).astype(np.float64)


def compute_common_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    probabilities_arr = np.asarray(probabilities, dtype=np.float64)
    resolved_labels = _coerce_labels(labels, probabilities_arr)
    resolved_class_names = list(class_names) if class_names is not None else [str(v) for v in resolved_labels]

    full_metrics = multiclass_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_proba=probabilities_arr,
        labels=resolved_labels,
    )
    report_text = format_classification_report(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_proba=probabilities_arr,
        labels=resolved_labels,
        class_names=resolved_class_names,
    )
    confidences = extract_softmax_confidences(probabilities_arr)

    return {
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "probabilities": probabilities_arr,
        "confidences": confidences,
        "accuracy": float(full_metrics["accuracy"]),
        "macro_f1": float(full_metrics["f1_macro"]),
        "weighted_f1": float(full_metrics["f1_weighted"]),
        "confusion_matrix": np.asarray(full_metrics["confusion_matrix"], dtype=np.int64),
        "classification_report": report_text,
        "metrics": full_metrics,
    }


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 0,
    scaler: Optional[FeatureStandardizer] = None,
    labels: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    already_scaled: bool = False,
) -> Dict[str, Any]:
    X_eval = np.asarray(X, dtype=np.float32)
    if not already_scaled and scaler is not None:
        X_eval = scaler.transform(X_eval).astype(np.float32)
    probabilities = predict_proba(
        model=model,
        X=X_eval,
        device=device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    y_pred = np.argmax(probabilities, axis=1).astype(np.int64) if len(probabilities) else np.zeros((0,), dtype=np.int64)
    return compute_common_metrics(
        y_true=np.asarray(y, dtype=np.int64),
        y_pred=y_pred,
        probabilities=probabilities,
        labels=labels,
        class_names=class_names,
    )


def _infer_feature_names(
    checkpoint: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> List[str]:
    checkpoint_names = checkpoint.get("feature_names")
    if isinstance(checkpoint_names, list) and checkpoint_names:
        return [str(name) for name in checkpoint_names]
    features_block = metrics.get("features", {})
    feature_names = features_block.get("feature_names", [])
    if isinstance(feature_names, list) and feature_names:
        return [str(name) for name in feature_names]
    raise FileNotFoundError("Could not recover feature names from checkpoint or metrics.json.")


def _infer_labels_and_classes(
    checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Tuple[List[int], List[str]]:
    raw_labels = checkpoint.get("labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raw_labels = list(metrics.get("validation", {}).get("per_class", {}).keys())
    labels = [int(label) for label in raw_labels] if raw_labels else sorted(int(v) for v in DEFAULT_PHASE_CODE_MAP.values())

    raw_class_names = checkpoint.get("class_names")
    if isinstance(raw_class_names, list) and len(raw_class_names) == len(labels):
        class_names = [str(name) for name in raw_class_names]
    else:
        phase_code_map = config.get("phase_code_map", DEFAULT_PHASE_CODE_MAP)
        inverse_phase_map = {int(v): str(k) for k, v in phase_code_map.items()}
        class_names = [inverse_phase_map.get(label, str(label)) for label in labels]
    return labels, class_names


def load_trained_artifacts(
    model_dir: str | Path,
    checkpoint_name: str = "best_model.pt",
    device: str = "auto",
) -> ArtifactBundle:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    checkpoint_path = model_path / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_json_file(model_path / "config.json")
    metrics = load_json_file(model_path / "metrics.json")
    efficiency = load_json_file(model_path / "efficiency.json")

    resolved_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

    feature_names = _infer_feature_names(checkpoint=checkpoint, metrics=metrics)
    labels, class_names = _infer_labels_and_classes(checkpoint=checkpoint, config=config, metrics=metrics)

    model_name = str(checkpoint.get("model_name") or config.get("model") or metrics.get("model_type") or "").strip()
    if not model_name:
        raise RuntimeError("Could not determine model type from checkpoint or config.")

    hidden_layers = config.get("hidden_layers_resolved")
    if not isinstance(hidden_layers, list):
        hidden_layers = parse_hidden_layers(str(config.get("hidden_layers", "64,32")))
    hidden_layers = [int(value) for value in hidden_layers]
    dropout = float(config.get("dropout", metrics.get("hyperparameters", {}).get("dropout", 0.0)))

    model = make_model(
        model_name=model_name,
        input_dim=len(feature_names),
        num_classes=max(labels) + 1,
        hidden_layers=hidden_layers,
        dropout=dropout,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = FeatureStandardizer()
    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std = checkpoint.get("scaler_std")
    if scaler_mean is None or scaler_std is None:
        raise RuntimeError(f"Checkpoint is missing scaler statistics: {checkpoint_path}")
    scaler.mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler.std = np.asarray(scaler_std, dtype=np.float64)

    return ArtifactBundle(
        model_dir=model_path,
        checkpoint_path=checkpoint_path,
        config=config,
        metrics=metrics,
        efficiency=efficiency,
        checkpoint=checkpoint,
        model_name=model_name,
        device=resolved_device,
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        labels=labels,
        class_names=class_names,
    )


def _load_split_subjects(model_dir: Path, config: Mapping[str, Any], groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    split_path = model_dir / "split_subjects.json"
    if split_path.exists():
        split_subjects = load_json_file(split_path)
        train_subjects = set(str(value) for value in split_subjects.get("train", []))
        val_subjects = set(str(value) for value in split_subjects.get("val", []))
        test_subjects = set(str(value) for value in split_subjects.get("test", []))
        train_idx = np.where(np.isin(groups.astype(str), list(train_subjects)))[0]
        val_idx = np.where(np.isin(groups.astype(str), list(val_subjects)))[0]
        test_idx = np.where(np.isin(groups.astype(str), list(test_subjects)))[0]
        if len(train_idx) and len(val_idx) and len(test_idx):
            return train_idx, val_idx, test_idx, {
                "train": sorted(train_subjects),
                "val": sorted(val_subjects),
                "test": sorted(test_subjects),
            }

    train_idx, val_idx, test_idx, split_subjects = split_train_val_test_by_subject(
        groups=groups,
        train_ratio=float(config.get("train_ratio", 0.7)),
        val_ratio=float(config.get("val_ratio", 0.15)),
        test_ratio=float(config.get("test_ratio", 0.15)),
        seed=int(config.get("seed", DEFAULT_RANDOM_SEED)),
    )
    return train_idx, val_idx, test_idx, split_subjects


def load_dataset_for_run(bundle: ArtifactBundle) -> DatasetBundle:
    config = bundle.config
    participants = config.get("participants", DEFAULT_PARTICIPANTS)
    if isinstance(participants, str):
        participants = [item.strip() for item in participants.split(",") if item.strip()]
    participants = [str(value) for value in participants]

    loader = EEGDataLoader(
        data_dir=config.get("data_dir", DEFAULT_DATA_DIR),
        participants=participants,
        phase_code_map=config.get("phase_code_map", DEFAULT_PHASE_CODE_MAP),
        json_phase_to_canonical=config.get("json_phase_to_canonical", DEFAULT_JSON_PHASE_TO_CANONICAL),
    )
    extractor = EEGFeatureExtractor(sampling_rate=int(config.get("sampling_rate", DEFAULT_SAMPLING_RATE)))
    X, y, groups = build_feature_table(loader, extractor)

    derived_feature_names = build_feature_names(extractor)
    if not bool(config.get("use_freq_features", True)):
        keep_idx = [idx for idx, name in enumerate(derived_feature_names) if not str(name).endswith("_power")]
        X = X[:, keep_idx]
        derived_feature_names = [derived_feature_names[idx] for idx in keep_idx]

    if derived_feature_names != bundle.feature_names:
        index_map = {name: idx for idx, name in enumerate(derived_feature_names)}
        missing = [name for name in bundle.feature_names if name not in index_map]
        if missing:
            raise RuntimeError(f"Saved feature schema does not match rebuilt dataset. Missing features: {missing}")
        X = X[:, [index_map[name] for name in bundle.feature_names]]
        derived_feature_names = list(bundle.feature_names)

    train_idx, val_idx, test_idx, split_subjects = _load_split_subjects(
        model_dir=bundle.model_dir,
        config=config,
        groups=groups,
    )

    return DatasetBundle(
        X=X.astype(np.float64),
        y=y.astype(np.int64),
        groups=groups,
        feature_names=derived_feature_names,
        split_subjects=split_subjects,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )


def profile_inference_if_available(
    bundle: ArtifactBundle,
    X: np.ndarray,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    warmup_batches: Optional[int] = None,
    already_scaled: bool = False,
) -> Dict[str, Any]:
    config = bundle.config
    inference_batch_size = int(batch_size or config.get("inference_batch_size") or config.get("batch_size", 256))
    if inference_batch_size <= 0:
        inference_batch_size = int(config.get("batch_size", 256))
    workers = int(num_workers if num_workers is not None else config.get("num_workers", 0))
    warmup = int(warmup_batches if warmup_batches is not None else config.get("warmup_batches", 3))

    X_profile = np.asarray(X, dtype=np.float32)
    if not already_scaled:
        X_profile = bundle.scaler.transform(X_profile).astype(np.float32)

    return profile_inference(
        model=bundle.model,
        X=X_profile,
        device=bundle.device,
        batch_size=inference_batch_size,
        num_workers=workers,
        warmup_batches=warmup,
    )


def summarize_feature_reference(
    X: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(feature_names))
    summary = pd.DataFrame({
        "feature": frame.columns,
        "mean": frame.mean(axis=0).values,
        "std": frame.std(axis=0).replace(0.0, 1.0).values,
        "min": frame.min(axis=0).values,
        "max": frame.max(axis=0).values,
        "q01": frame.quantile(0.01).values,
        "q99": frame.quantile(0.99).values,
    })
    return summary


def save_summary_bundle(
    output_dir: str | Path,
    summary: Optional[Mapping[str, Any]] = None,
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
    text_blobs: Optional[Mapping[str, str]] = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary is not None:
        save_json(out_dir / "summary.json", dict(summary))

    for name, table in (tables or {}).items():
        save_csv(out_dir / f"{name}.csv", table)

    for name, text in (text_blobs or {}).items():
        file_path = out_dir / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")


def select_key_features(
    X: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 5,
) -> List[str]:
    frame = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(feature_names))
    variability = frame.std(axis=0).sort_values(ascending=False)
    return [str(name) for name in variability.head(int(top_k)).index.tolist()]


def ensure_non_empty(name: str, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
    if X is None or len(X) == 0:
        raise ValueError(f"{name} input is empty.")
    if y is not None and len(y) != len(X):
        raise ValueError(f"{name} label count does not match feature rows.")


def confidence_threshold_predictions(probabilities: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    confidences = extract_softmax_confidences(probabilities)
    if np.asarray(probabilities).size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=bool)
    predicted = np.argmax(probabilities, axis=1).astype(np.int64)
    accepted_mask = confidences >= float(threshold)
    return predicted, accepted_mask


def flatten_classification_report(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flattened.update(flatten_classification_report(value, prefix=f"{name}_"))
        else:
            flattened[name] = value
    return flattened


from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import psutil
except Exception:
    psutil = None

from app import (
    DATA_DIR as DEFAULT_DATA_DIR,
    JSON_PHASE_TO_CANONICAL as DEFAULT_JSON_PHASE_TO_CANONICAL,
    PARTICIPANTS as DEFAULT_PARTICIPANTS,
    PHASE_CODE_MAP as DEFAULT_PHASE_CODE_MAP,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
    SAMPLING_RATE as DEFAULT_SAMPLING_RATE,
    TORCH_LINEAR_CONFIG as DEFAULT_LINEAR_CONFIG,
    TORCH_MLP_CONFIG as DEFAULT_MLP_CONFIG,
    EEGDataLoader,
    EEGFeatureExtractor,
    FeatureStandardizer,
    LinearClassifier,
    MLPClassifier,
    build_feature_table,
    plot_confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    mlp_hidden_default = ",".join(str(v) for v in DEFAULT_MLP_CONFIG.get("hidden_dims", [64, 32]))
    parser = argparse.ArgumentParser(
        description="Milestone-2 EEG phase training with participant-wise train/val/test split."
    )
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--participants", type=str, default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--model", type=str, choices=["torch_linear", "torch_mlp"], default="torch_linear")
    parser.add_argument("--hidden_layers", type=str, default=mlp_hidden_default)
    parser.add_argument("--dropout", type=float, default=float(DEFAULT_MLP_CONFIG.get("dropout", 0.2)))
    parser.add_argument("--lr", type=float, default=float(DEFAULT_LINEAR_CONFIG.get("learning_rate", 1e-3)))
    parser.add_argument("--weight_decay", type=float, default=float(DEFAULT_LINEAR_CONFIG.get("weight_decay", 1e-4)))
    parser.add_argument("--batch_size", type=int, default=int(DEFAULT_LINEAR_CONFIG.get("batch_size", 256)))
    parser.add_argument("--epochs", type=int, default=int(DEFAULT_LINEAR_CONFIG.get("epochs", 80)))
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=int(DEFAULT_RANDOM_SEED))
    parser.add_argument("--sampling_rate", type=int, default=int(DEFAULT_SAMPLING_RATE))
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--inference_batch_size", type=int, default=0)
    parser.add_argument("--warmup_batches", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_freq_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_participants(raw: str) -> List[str]:
    values = [v.strip() for v in raw.split(",")]
    return [v for v in values if v]


def parse_hidden_layers(raw: str) -> List[int]:
    items = [v.strip() for v in raw.split(",") if v.strip()]
    if not items:
        return [64, 32]
    dims: List[int] = []
    for item in items:
        value = int(item)
        if value <= 0:
            raise ValueError(f"Hidden layer size must be > 0, got: {value}")
        dims.append(value)
    return dims


def set_full_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_roc_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = y_true_binary.astype(np.int64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        return None
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def binary_pr_auc(y_true_binary: np.ndarray, scores: np.ndarray) -> Optional[float]:
    y = y_true_binary.astype(np.int64)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores):
        return None
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    area = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return area


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: Sequence[int],
) -> Dict[str, object]:
    label_list = [int(v) for v in labels]
    num_classes = len(label_list)
    label_to_index = {label: i for i, label in enumerate(label_list)}

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_t, y_p in zip(y_true.tolist(), y_pred.tolist()):
        if int(y_t) in label_to_index and int(y_p) in label_to_index:
            cm[label_to_index[int(y_t)], label_to_index[int(y_p)]] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)
    total = float(np.sum(cm))

    precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_per_class = np.divide(
        2.0 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) > 0,
    )

    precision_macro = float(np.mean(precision_per_class)) if num_classes else 0.0
    recall_macro = float(np.mean(recall_per_class)) if num_classes else 0.0
    f1_macro = float(np.mean(f1_per_class)) if num_classes else 0.0

    precision_weighted = safe_div(float(np.sum(precision_per_class * support)), float(np.sum(support)))
    recall_weighted = safe_div(float(np.sum(recall_per_class * support)), float(np.sum(support)))
    f1_weighted = safe_div(float(np.sum(f1_per_class * support)), float(np.sum(support)))

    tp_total = float(np.sum(tp))
    fp_total = float(np.sum(fp))
    fn_total = float(np.sum(fn))
    precision_micro = safe_div(tp_total, tp_total + fp_total)
    recall_micro = safe_div(tp_total, tp_total + fn_total)
    f1_micro = safe_div(2.0 * precision_micro * recall_micro, precision_micro + recall_micro)
    accuracy = safe_div(tp_total, total)

    auc_values: List[Tuple[float, float]] = []
    pr_values: List[Tuple[float, float]] = []
    for idx, label in enumerate(label_list):
        if idx >= y_proba.shape[1]:
            continue
        y_binary = (y_true == label).astype(np.int64)
        class_support = float(np.sum(y_binary))
        auc = binary_roc_auc(y_binary, y_proba[:, idx])
        pr_auc = binary_pr_auc(y_binary, y_proba[:, idx])
        if auc is not None:
            auc_values.append((auc, class_support))
        if pr_auc is not None:
            pr_values.append((pr_auc, class_support))

    auroc_macro = float(np.mean([v for v, _ in auc_values])) if auc_values else None
    auroc_weighted = (
        float(np.sum([v * w for v, w in auc_values]) / np.sum([w for _, w in auc_values])) if auc_values else None
    )
    pr_auc_macro = float(np.mean([v for v, _ in pr_values])) if pr_values else None
    pr_auc_weighted = (
        float(np.sum([v * w for v, w in pr_values]) / np.sum([w for _, w in pr_values])) if pr_values else None
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_list):
        per_class[str(label)] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1": float(f1_per_class[idx]),
            "support": int(support[idx]),
        }

    return {
        "accuracy": float(accuracy),
        "precision_macro": precision_macro,
        "precision_micro": float(precision_micro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": recall_macro,
        "recall_micro": float(recall_micro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": f1_macro,
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "auroc_ovr_macro": auroc_macro,
        "auroc_ovr_weighted": auroc_weighted,
        "pr_auc_ovr_macro": pr_auc_macro,
        "pr_auc_ovr_weighted": pr_auc_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: Sequence[int],
    class_names: Sequence[str],
) -> str:
    metrics = multiclass_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, labels=labels)
    lines: List[str] = []
    lines.append(f"{'class':<16}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}")
    lines.append("")
    for label, class_name in zip(labels, class_names):
        row = metrics["per_class"][str(int(label))]
        lines.append(
            f"{class_name:<16}{row['precision']:>10.4f}{row['recall']:>10.4f}{row['f1']:>10.4f}{int(row['support']):>10d}"
        )

    total_support = int(len(y_true))
    lines.append("")
    lines.append(f"{'accuracy':<16}{'':>10}{'':>10}{metrics['accuracy']:>10.4f}{total_support:>10d}")
    lines.append(
        f"{'macro avg':<16}{metrics['precision_macro']:>10.4f}{metrics['recall_macro']:>10.4f}{metrics['f1_macro']:>10.4f}{total_support:>10d}"
    )
    lines.append(
        f"{'micro avg':<16}{metrics['precision_micro']:>10.4f}{metrics['recall_micro']:>10.4f}{metrics['f1_micro']:>10.4f}{total_support:>10d}"
    )
    lines.append(
        f"{'weighted avg':<16}{metrics['precision_weighted']:>10.4f}{metrics['recall_weighted']:>10.4f}{metrics['f1_weighted']:>10.4f}{total_support:>10d}"
    )
    lines.append("")
    lines.append(
        f"AUROC OvR macro={metrics['auroc_ovr_macro'] if metrics['auroc_ovr_macro'] is not None else 'NA'} "
        f"weighted={metrics['auroc_ovr_weighted'] if metrics['auroc_ovr_weighted'] is not None else 'NA'}"
    )
    lines.append(
        f"PR-AUC OvR macro={metrics['pr_auc_ovr_macro'] if metrics['pr_auc_ovr_macro'] is not None else 'NA'} "
        f"weighted={metrics['pr_auc_ovr_weighted'] if metrics['pr_auc_ovr_weighted'] is not None else 'NA'}"
    )
    return "\n".join(lines) + "\n"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def split_train_val_test_by_subject(
    groups: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)
    if n_subjects < 3:
        raise RuntimeError("Need at least 3 participants for subject-wise train/val/test split.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise RuntimeError("train_ratio + val_ratio + test_ratio must be > 0.")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    raw_counts = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64) * float(n_subjects)
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_subjects - int(np.sum(counts)))
    fractional = raw_counts - counts
    order = np.argsort(-fractional, kind="mergesort")
    for idx in order[:remainder]:
        counts[int(idx)] += 1

    # Ensure non-empty train/val/test splits when possible (n_subjects >= 3 checked above).
    for target_idx in [0, 1, 2]:
        if counts[target_idx] >= 1:
            continue
        donor_order = np.argsort(-counts, kind="mergesort")
        donated = False
        for donor_idx in donor_order.tolist():
            if donor_idx != target_idx and counts[donor_idx] > 1:
                counts[donor_idx] -= 1
                counts[target_idx] += 1
                donated = True
                break
        if not donated:
            raise RuntimeError(
                "Could not create non-empty train/val/test subject splits. Increase participant count or adjust ratios."
            )

    n_train, n_val, n_test = int(counts[0]), int(counts[1]), int(counts[2])

    train_subjects = sorted(str(v) for v in shuffled[:n_train].tolist())
    val_subjects = sorted(str(v) for v in shuffled[n_train : n_train + n_val].tolist())
    test_subjects = sorted(str(v) for v in shuffled[n_train + n_val :].tolist())

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Subject leakage detected in train/val/test split.")

    train_idx = np.where(np.array([str(g) in train_set for g in groups], dtype=bool))[0]
    val_idx = np.where(np.array([str(g) in val_set for g in groups], dtype=bool))[0]
    test_idx = np.where(np.array([str(g) in test_set for g in groups], dtype=bool))[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Train/val/test split produced an empty sample partition.")

    split_subjects = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }
    return train_idx, val_idx, test_idx, split_subjects


def compute_class_weights(y_train: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    total = float(np.sum(counts))
    weights = np.zeros(num_classes, dtype=np.float32)
    for idx, count in enumerate(counts):
        if count > 0:
            weights[idx] = float(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def make_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    hidden_layers: List[int],
    dropout: float,
) -> nn.Module:
    if model_name == "torch_linear":
        return LinearClassifier(input_dim=input_dim, num_classes=num_classes)
    if model_name == "torch_mlp":
        return MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            bs = int(y_batch.size(0))
            total_loss += float(loss.item()) * bs
            n_samples += bs
            all_preds.append(preds.astype(np.int64))
            all_probs.append(probs.astype(np.float64))
    avg_loss = total_loss / max(1, n_samples)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
    y_proba = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 0), dtype=np.float64)
    return avg_loss, y_pred, y_proba


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    X_tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(
        X_tensor,
        batch_size=min(batch_size, max(1, len(X_tensor))),
        shuffle=False,
        num_workers=num_workers,
    )
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch in loader:
            logits = model(X_batch.to(device))
            probs = torch.softmax(logits, dim=1)
            chunks.append(probs.cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(chunks).astype(np.float64)


def weighted_percentile(values: List[float], weights: List[int], q: float) -> float:
    if not values:
        return 0.0
    arr_v = np.array(values, dtype=np.float64)
    arr_w = np.array(weights, dtype=np.float64)
    order = np.argsort(arr_v)
    arr_v = arr_v[order]
    arr_w = arr_w[order]
    cum = np.cumsum(arr_w)
    threshold = q * float(np.sum(arr_w))
    idx = int(np.searchsorted(cum, threshold, side="left"))
    idx = max(0, min(idx, len(arr_v) - 1))
    return float(arr_v[idx])


def profile_inference(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    warmup_batches: int,
) -> Dict[str, Optional[float]]:
    model.eval()
    X_tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(
        X_tensor,
        batch_size=min(batch_size, max(1, len(X_tensor))),
        shuffle=False,
        num_workers=num_workers,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    it = iter(loader)
    with torch.no_grad():
        for _ in range(max(0, warmup_batches)):
            try:
                batch = next(it)
            except StopIteration:
                break
            _ = model(batch.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    latency_values_ms: List[float] = []
    latency_weights: List[int] = []
    total_samples = 0
    start_time = time.perf_counter()
    peak_ram = current_rss_mb()

    with torch.no_grad():
        for batch in loader:
            if peak_ram is not None:
                rss = current_rss_mb()
                if rss is not None:
                    peak_ram = rss if peak_ram is None else max(peak_ram, rss)

            batch = batch.to(device)
            bs = int(batch.size(0))
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            elapsed = time.perf_counter() - t0
            per_sample_ms = (elapsed / max(1, bs)) * 1000.0
            latency_values_ms.append(float(per_sample_ms))
            latency_weights.append(bs)
            total_samples += bs

    total_time = time.perf_counter() - start_time
    p50 = weighted_percentile(latency_values_ms, latency_weights, 0.50)
    p90 = weighted_percentile(latency_values_ms, latency_weights, 0.90)
    throughput = safe_div(float(total_samples), float(total_time))
    peak_vram = None
    if device.type == "cuda":
        peak_vram = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

    return {
        "latency_per_sample_ms_p50": float(p50),
        "latency_per_sample_ms_p90": float(p90),
        "throughput_samples_per_sec": float(throughput),
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
        "n_samples_profiled": int(total_samples),
        "warmup_batches": int(max(0, warmup_batches)),
    }


def create_run_dir(runs_dir: Path, model: str, tag: str) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag.strip() or "default").strip("-") or "default"
    name = f"{timestamp}_{model}_{safe_tag}"
    out_dir = runs_dir / name
    suffix = 1
    while out_dir.exists():
        out_dir = runs_dir / f"{name}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir, timestamp


def save_learning_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    csv_path = out_dir / "learning_curves.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "epoch_time_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    val_macro_f1 = [float(row["val_macro_f1"]) for row in history]
    val_weighted_f1 = [float(row["val_weighted_f1"]) for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_macro_f1, label="val_macro_f1")
    axes[1].plot(epochs, val_weighted_f1, label="val_weighted_f1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Validation F1 Curve")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def build_feature_names(extractor: EEGFeatureExtractor, window_size: int = 1000) -> List[str]:
    dummy = np.zeros((1, window_size), dtype=np.float32)
    return list(extractor.extract_features(dummy).keys())


def main() -> None:
    args = parse_args()
    participants = parse_participants(args.participants)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    set_full_seed(args.seed)

    if len(participants) < 3:
        raise RuntimeError("Please provide at least 3 participants for train/val/test splitting.")

    runs_dir = Path(args.runs_dir)
    out_dir, run_timestamp = create_run_dir(runs_dir=runs_dir, model=args.model, tag=args.tag)

    device = resolve_device(args.device)
    print("=" * 72)
    print("Milestone-2 EEG Training")
    print("=" * 72)
    print(f"Run directory: {out_dir}")
    print(f"Device: {device}")

    loader = EEGDataLoader(
        data_dir=args.data_dir,
        participants=participants,
        phase_code_map=DEFAULT_PHASE_CODE_MAP,
        json_phase_to_canonical=DEFAULT_JSON_PHASE_TO_CANONICAL,
    )
    extractor = EEGFeatureExtractor(sampling_rate=args.sampling_rate)
    X, y, groups = build_feature_table(loader, extractor)

    feature_names = build_feature_names(extractor)
    if X.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature shape mismatch. X has {X.shape[1]} columns, expected {len(feature_names)} from extractor."
        )

    if not args.use_freq_features:
        keep_idx = [i for i, name in enumerate(feature_names) if not name.endswith("_power")]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]

    train_idx, val_idx, test_idx, split_subjects = split_train_val_test_by_subject(
        groups=groups,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    train_subjects = set(split_subjects["train"])
    val_subjects = set(split_subjects["val"])
    test_subjects = set(split_subjects["test"])
    if train_subjects & val_subjects or train_subjects & test_subjects or val_subjects & test_subjects:
        raise RuntimeError("Subject leakage check failed.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Total samples: {len(X)}")
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"Subjects train={split_subjects['train']}")
    print(f"Subjects val={split_subjects['val']}")
    print(f"Subjects test={split_subjects['test']}")

    scaler = FeatureStandardizer()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    labels = sorted(int(v) for v in DEFAULT_PHASE_CODE_MAP.values())
    inverse_phase_map = {int(v): k for k, v in DEFAULT_PHASE_CODE_MAP.items()}
    class_names = [inverse_phase_map.get(lbl, str(lbl)) for lbl in labels]
    num_classes = max(labels) + 1

    model = make_model(
        model_name=args.model,
        input_dim=int(X_train_scaled.shape[1]),
        num_classes=int(num_classes),
        hidden_layers=hidden_layers,
        dropout=float(args.dropout),
    ).to(device)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(y_train=y_train, num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_scaled),
        torch.from_numpy(y_val.astype(np.int64)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(int(args.batch_size), max(1, len(train_dataset))),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(int(args.batch_size), max(1, len(val_dataset))),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    best_model_path = out_dir / "best_model.pt"
    last_model_path = out_dir / "last_model.pt"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    training_start = time.perf_counter()
    training_peak_ram = current_rss_mb()
    best_epoch = -1
    best_val_macro_f1 = -np.inf
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []
    early_stopped = False
    epochs_ran = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        seen = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            bs = int(y_batch.size(0))
            running_loss += float(loss.item()) * bs
            seen += bs

            if training_peak_ram is not None:
                rss = current_rss_mb()
                if rss is not None:
                    training_peak_ram = max(training_peak_ram, rss)

        train_loss = running_loss / max(1, seen)
        val_loss, val_pred, val_proba = evaluate_model(model=model, loader=val_loader, criterion=criterion, device=device)
        val_metrics = multiclass_metrics(y_true=y_val, y_pred=val_pred, y_proba=val_proba, labels=labels)
        epoch_time = time.perf_counter() - epoch_start
        epochs_ran = epoch

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["f1_macro"]),
            "val_weighted_f1": float(val_metrics["f1_weighted"]),
            "epoch_time_sec": float(epoch_time),
        }
        history.append(row)

        score = float(val_metrics["f1_macro"])
        improved = score > (best_val_macro_f1 + 1e-8)
        if improved:
            best_val_macro_f1 = score
            best_epoch = epoch
            no_improve_epochs = 0
            checkpoint = {
                "epoch": int(epoch),
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean.tolist() if scaler.mean is not None else None,
                "scaler_std": scaler.std.tolist() if scaler.std is not None else None,
                "best_val_macro_f1": float(best_val_macro_f1),
                "feature_names": feature_names,
                "labels": labels,
                "class_names": class_names,
            }
            torch.save(checkpoint, best_model_path)
        else:
            no_improve_epochs += 1

        if args.verbose:
            print(
                f"Epoch {epoch:03d}/{int(args.epochs)} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['f1_macro']:.4f}"
            )

        if no_improve_epochs >= int(args.patience):
            early_stopped = True
            if args.verbose:
                print(f"Early stopping triggered at epoch {epoch} (patience={args.patience}).")
            break

    training_total_time = time.perf_counter() - training_start
    training_peak_vram = None
    if device.type == "cuda":
        training_peak_vram = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

    last_checkpoint = {
        "epoch": int(epochs_ran),
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean.tolist() if scaler.mean is not None else None,
        "scaler_std": scaler.std.tolist() if scaler.std is not None else None,
        "best_val_macro_f1": float(best_val_macro_f1),
        "feature_names": feature_names,
        "labels": labels,
        "class_names": class_names,
    }
    torch.save(last_checkpoint, last_model_path)

    if not best_model_path.exists():
        torch.save(last_checkpoint, best_model_path)
        best_epoch = epochs_ran
        best_val_macro_f1 = float(history[-1]["val_macro_f1"]) if history else 0.0

    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_proba_best = predict_proba(
        model=model,
        X=X_val_scaled,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    val_pred_best = np.argmax(val_proba_best, axis=1).astype(np.int64)
    val_metrics_final = multiclass_metrics(y_true=y_val, y_pred=val_pred_best, y_proba=val_proba_best, labels=labels)

    test_proba = predict_proba(
        model=model,
        X=X_test_scaled,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )
    test_pred = np.argmax(test_proba, axis=1).astype(np.int64)
    test_metrics = multiclass_metrics(y_true=y_test, y_pred=test_pred, y_proba=test_proba, labels=labels)

    save_learning_curves(history=history, out_dir=out_dir)

    cm_fig = plot_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        labels=labels,
        class_names=class_names,
        title="Confusion Matrix (Test)",
    )
    cm_fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(cm_fig.gcf())

    report_text = []
    report_text.append("Validation Classification Report (best model)\n")
    report_text.append(
        format_classification_report(
            y_true=y_val,
            y_pred=val_pred_best,
            y_proba=val_proba_best,
            labels=labels,
            class_names=class_names,
        )
    )
    report_text.append("\nTest Classification Report (best model)\n")
    report_text.append(
        format_classification_report(
            y_true=y_test,
            y_pred=test_pred,
            y_proba=test_proba,
            labels=labels,
            class_names=class_names,
        )
    )
    (out_dir / "classification_report.txt").write_text("\n".join(report_text), encoding="utf-8")

    (out_dir / "split_subjects.json").write_text(json.dumps(split_subjects, indent=2), encoding="utf-8")

    params_count = int(sum(p.numel() for p in model.parameters()))
    best_ckpt_size = int(best_model_path.stat().st_size) if best_model_path.exists() else 0
    last_ckpt_size = int(last_model_path.stat().st_size) if last_model_path.exists() else 0
    epoch_times = [float(row["epoch_time_sec"]) for row in history]

    efficiency_payload = {
        "training": {
            "total_training_time_sec": float(training_total_time),
            "epochs_ran": int(epochs_ran),
            "time_per_epoch_sec_mean": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "time_per_epoch_sec": epoch_times,
            "early_stopped": bool(early_stopped),
            "best_epoch": int(best_epoch),
            "peak_ram_mb": training_peak_ram,
            "peak_vram_mb": training_peak_vram,
            "parameter_count": params_count,
            "checkpoint_size_bytes": {
                "best_model.pt": best_ckpt_size,
                "last_model.pt": last_ckpt_size,
            },
        },
        "inference": {},
    }

    if args.profile:
        inference_batch_size = int(args.inference_batch_size) if int(args.inference_batch_size) > 0 else int(args.batch_size)
        efficiency_payload["inference"] = profile_inference(
            model=model,
            X=X_test_scaled,
            device=device,
            batch_size=inference_batch_size,
            num_workers=int(args.num_workers),
            warmup_batches=int(args.warmup_batches),
        )
        efficiency_payload["inference"]["batch_size"] = int(inference_batch_size)
    else:
        efficiency_payload["inference"] = {"enabled": False}

    (out_dir / "efficiency.json").write_text(json.dumps(efficiency_payload, indent=2), encoding="utf-8")

    metrics_payload = {
        "timestamp": run_timestamp,
        "model_type": args.model,
        "hyperparameters": {
            "hidden_layers": hidden_layers if args.model == "torch_mlp" else [],
            "dropout": float(args.dropout) if args.model == "torch_mlp" else 0.0,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "use_class_weights": bool(args.use_class_weights),
        },
        "features": {
            "use_freq_features": bool(args.use_freq_features),
            "n_features": int(len(feature_names)),
            "feature_names": feature_names,
        },
        "seed": int(args.seed),
        "split_subjects": split_subjects,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "dataset": {
            "n_total": int(len(X)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "label_distribution_train": {str(int(k)): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "label_distribution_val": {str(int(k)): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "label_distribution_test": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "validation": val_metrics_final,
        "test": test_metrics,
        "early_stopping": {
            "enabled": True,
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val_macro_f1),
            "stopped_early": bool(early_stopped),
            "epochs_ran": int(epochs_ran),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    resolved_config = vars(args).copy()
    resolved_config["participants"] = participants
    resolved_config["hidden_layers_resolved"] = hidden_layers
    resolved_config["phase_code_map"] = DEFAULT_PHASE_CODE_MAP
    resolved_config["json_phase_to_canonical"] = DEFAULT_JSON_PHASE_TO_CANONICAL
    resolved_config["resolved_device"] = str(device)
    resolved_config["run_dir"] = str(out_dir)
    (out_dir / "config.json").write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

    print("\nRun complete.")
    print(f"Best epoch: {best_epoch} | best val macro-F1: {best_val_macro_f1:.4f}")
    print(f"Validation macro-F1: {val_metrics_final['f1_macro']:.4f}")
    print(f"Test macro-F1: {test_metrics['f1_macro']:.4f}")
    print(f"Artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()

# EEG baseline training from `sub_*.json` files.
# The JSON files already contain segmented EEG arrays with phase labels in keys.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from scipy.stats import rankdata
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import configuration
import sys

sys.path.insert(0, str(Path(__file__).parent))
try:
    import config as project_config

    RANDOM_SEED = int(getattr(project_config, "RANDOM_SEED", 42))
    DATA_DIR = str(getattr(project_config, "DATA_DIR", Path.cwd() / "EEG data"))
    PARTICIPANTS = list(getattr(project_config, "PARTICIPANTS", ["P2", "P3", "P4"]))
    SAMPLING_RATE = int(getattr(project_config, "SAMPLING_RATE", 500))
    WINDOW_SIZE = int(getattr(project_config, "WINDOW_SIZE", 1000))
    WINDOW_OVERLAP = float(getattr(project_config, "WINDOW_OVERLAP", 0.5))
    TEST_SIZE = float(getattr(project_config, "TEST_SIZE", 0.3))
    RESULTS_DIR = str(getattr(project_config, "RESULTS_DIR", Path.cwd() / "results"))
    SAVE_PLOTS = bool(getattr(project_config, "SAVE_PLOTS", True))
    PLOT_FORMAT = str(getattr(project_config, "PLOT_FORMAT", "png"))
    PLOT_DPI = int(getattr(project_config, "PLOT_DPI", 150))
    TORCH_LINEAR_CONFIG = dict(
        getattr(
            project_config,
            "TORCH_LINEAR_CONFIG",
            {
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 256,
                "epochs": 80,
                "device": "auto",
                "verbose": True,
            },
        )
    )
    TORCH_MLP_CONFIG = dict(
        getattr(
            project_config,
            "TORCH_MLP_CONFIG",
            {
                "hidden_dims": [64, 32],
                "dropout": 0.2,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 256,
                "epochs": 120,
                "device": "auto",
                "verbose": True,
            },
        )
    )
    MODEL_VARIANTS = list(getattr(project_config, "MODEL_VARIANTS", ["torch_linear", "torch_mlp"]))
    PHASE_CODE_MAP = dict(
        getattr(
            project_config,
            "PHASE_CODE_MAP",
            {
                "RST": 0,
                "IDG": 1,
                "IDE": 2,
                "IDR": 3,
            },
        )
    )
    JSON_PHASE_TO_CANONICAL = dict(
        getattr(
            project_config,
            "JSON_PHASE_TO_CANONICAL",
            {
                "rest": "RST",
                "idea generation": "IDG",
                "idea evolution": "IDE",
                "idea rating": "IDR",
            },
        )
    )
except Exception:
    RANDOM_SEED = 42
    DATA_DIR = "d:/Zhu/AI Project/EEG data"
    PARTICIPANTS = ["P2", "P3", "P4"]
    SAMPLING_RATE = 500
    WINDOW_SIZE = 1000
    WINDOW_OVERLAP = 0.5
    TEST_SIZE = 0.3
    RESULTS_DIR = "d:/Zhu/AI Project/results"
    SAVE_PLOTS = True
    PLOT_FORMAT = "png"
    PLOT_DPI = 150
    TORCH_LINEAR_CONFIG = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 80,
        "device": "auto",
        "verbose": True,
    }
    TORCH_MLP_CONFIG = {
        "hidden_dims": [64, 32],
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "epochs": 120,
        "device": "auto",
        "verbose": True,
    }
    MODEL_VARIANTS = ["torch_linear", "torch_mlp"]
    PHASE_CODE_MAP = {"RST": 0, "IDG": 1, "IDE": 2, "IDR": 3}
    JSON_PHASE_TO_CANONICAL = {
        "rest": "RST",
        "idea generation": "IDG",
        "idea evolution": "IDE",
        "idea rating": "IDR",
    }


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureStandardizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class EEGDataLoader:
    """Load segmented EEG arrays directly from sub_*.json files."""

    def __init__(
        self,
        data_dir: str | Path,
        participants: List[str],
        phase_code_map: Dict[str, int],
        json_phase_to_canonical: Dict[str, str],
    ):
        self.data_dir = Path(data_dir)
        self.participants = participants
        self.phase_code_map = {k.upper(): int(v) for k, v in phase_code_map.items()}
        self.phase_aliases = {k.strip().lower(): v.strip().upper() for k, v in json_phase_to_canonical.items()}

    def participant_json_path(self, participant_id: str) -> Path:
        if not participant_id.upper().startswith("P") or not participant_id[1:].isdigit():
            raise ValueError(f"Unsupported participant ID format: {participant_id}")
        num = int(participant_id[1:])
        folder = self.data_dir / f"Participant-{num}"
        return folder / f"sub_{num:02d}.json"

    def load_json(self, participant_id: str) -> Dict[str, object]:
        path = self.participant_json_path(participant_id)
        if not path.exists():
            raise FileNotFoundError(f"Missing JSON file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def parse_segment_key(self, key: str) -> Tuple[Optional[int], Optional[str], str]:
        """Parse keys like '1_idea generation' -> (1, 'IDG', 'idea generation')."""
        raw = key.strip()
        cycle: Optional[int] = None
        phase_part = raw

        if "_" in raw:
            maybe_cycle, maybe_phase = raw.split("_", 1)
            if maybe_cycle.isdigit():
                cycle = int(maybe_cycle)
                phase_part = maybe_phase

        normalized = phase_part.strip().lower().replace("_", " ")
        canonical = self.phase_aliases.get(normalized)

        # Fallback fuzzy matching for slight naming variations.
        if canonical is None:
            for alias, canonical_name in self.phase_aliases.items():
                if alias in normalized:
                    canonical = canonical_name
                    break

        if canonical is None or canonical not in self.phase_code_map:
            return cycle, None, normalized

        return cycle, canonical, normalized


class EEGFeatureExtractor:
    """Extract simple time + frequency-domain features from EEG windows."""

    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100),
        }

    def compute_band_power(self, signal_data: np.ndarray, band_name: str) -> float:
        if band_name not in self.freq_bands:
            raise ValueError(f"Unknown band: {band_name}")

        freqs = rfftfreq(len(signal_data), d=1.0 / self.sampling_rate)
        fft_values = np.abs(rfft(signal_data))
        low_freq, high_freq = self.freq_bands[band_name]
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        if not np.any(band_mask):
            return 0.0
        return float(np.sum(fft_values[band_mask] ** 2))

    def extract_features(self, window_data: np.ndarray) -> Dict[str, float]:
        # Expected window_data shape: (n_channels, n_samples).
        if window_data.ndim != 2:
            raise ValueError("window_data must have shape (n_channels, n_samples)")

        # Phase-1 baseline uses channel-average features.
        signal_1d = window_data.mean(axis=0)
        features: Dict[str, float] = {
            "mean": float(np.mean(signal_1d)),
            "std": float(np.std(signal_1d)),
            "var": float(np.var(signal_1d)),
            "max": float(np.max(signal_1d)),
            "min": float(np.min(signal_1d)),
            "rms": float(np.sqrt(np.mean(signal_1d ** 2))),
        }

        for band_name in self.freq_bands:
            features[f"{band_name}_power"] = self.compute_band_power(signal_1d, band_name)

        return features


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = int(hidden)
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchEEGModel:
    """PyTorch baseline model wrapper for linear or MLP classifiers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        variant: str,
        model_config: Optional[Dict[str, object]] = None,
    ):
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.variant = variant

        config = dict(model_config or {})
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.batch_size = int(config.get("batch_size", 256))
        self.epochs = int(config.get("epochs", 80))
        self.verbose = bool(config.get("verbose", True))

        requested_device = str(config.get("device", "auto")).strip().lower()
        if requested_device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = requested_device
        self.device = torch.device(device_name)

        if variant == "linear":
            self.model = LinearClassifier(self.input_dim, self.num_classes)
        elif variant == "mlp":
            hidden_dims = [int(v) for v in config.get("hidden_dims", [64, 32])]
            dropout = float(config.get("dropout", 0.2))
            self.model = MLPClassifier(self.input_dim, self.num_classes, hidden_dims=hidden_dims, dropout=dropout)
        else:
            raise ValueError(f"Unsupported model variant: {variant}")

        self.model = self.model.to(self.device)
        self.scaler = FeatureStandardizer()
        self.is_trained = False

    def _compute_class_weights(self, y_train: np.ndarray) -> torch.Tensor:
        counts = np.bincount(y_train, minlength=self.num_classes).astype(np.float64)
        total = float(np.sum(counts))
        weights = np.zeros(self.num_classes, dtype=np.float32)
        for idx, count in enumerate(counts):
            if count > 0:
                weights[idx] = float(total / (self.num_classes * count))
            else:
                weights[idx] = 0.0
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        y_train_int = y_train.astype(np.int64)

        dataset = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train_int),
        )
        batch_size = min(self.batch_size, max(1, len(dataset)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        class_weights = self._compute_class_weights(y_train_int)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_seen = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                current_bs = int(y_batch.size(0))
                epoch_loss += float(loss.item()) * current_bs
                n_seen += current_bs

            if self.verbose and ((epoch + 1) in {1, self.epochs} or (epoch + 1) % 10 == 0):
                avg_loss = epoch_loss / max(1, n_seen)
                print(f"  Epoch {epoch + 1:03d}/{self.epochs} - loss: {avg_loss:.4f}")

        self.is_trained = True

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)
        X_tensor = torch.from_numpy(X_test_scaled)
        loader = DataLoader(X_tensor, batch_size=min(self.batch_size, max(1, len(X_tensor))), shuffle=False)

        probabilities: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)
                probabilities.append(probs.cpu().numpy())
        return np.vstack(probabilities)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=1).astype(np.int64)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, Optional[float]]:
        if y_pred is None:
            y_pred = self.predict(X_test)
        if y_proba is None:
            y_proba = self.predict_proba(X_test)
        return compute_metrics(y_test, y_pred, y_proba)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> Tuple[float, float, float, int]:
    y_t = y_true == label
    y_p = y_pred == label
    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum((~y_t) & y_p))
    fn = int(np.sum(y_t & (~y_p)))
    support = int(np.sum(y_t))
    precision = _safe_div(float(tp), float(tp + fp))
    recall = _safe_div(float(tp), float(tp + fn))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1, support


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    class_names: List[str],
) -> str:
    rows: List[Tuple[str, float, float, float, int]] = []
    total_support = int(len(y_true))

    for label, name in zip(labels, class_names):
        p, r, f1, support = _per_class_metrics(y_true, y_pred, int(label))
        rows.append((name, p, r, f1, support))

    macro_precision = float(np.mean([r[1] for r in rows])) if rows else 0.0
    macro_recall = float(np.mean([r[2] for r in rows])) if rows else 0.0
    macro_f1 = float(np.mean([r[3] for r in rows])) if rows else 0.0
    weighted_precision = _safe_div(sum(r[1] * r[4] for r in rows), float(total_support))
    weighted_recall = _safe_div(sum(r[2] * r[4] for r in rows), float(total_support))
    weighted_f1 = _safe_div(sum(r[3] * r[4] for r in rows), float(total_support))
    accuracy = _safe_div(float(np.sum(y_true == y_pred)), float(total_support))

    header = f"{'class':<16}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines = [header, ""]
    for name, p, r, f1, support in rows:
        lines.append(f"{name:<16}{p:>10.4f}{r:>10.4f}{f1:>10.4f}{support:>10d}")

    lines.append("")
    lines.append(f"{'accuracy':<16}{'':>10}{'':>10}{accuracy:>10.4f}{total_support:>10d}")
    lines.append(f"{'macro avg':<16}{macro_precision:>10.4f}{macro_recall:>10.4f}{macro_f1:>10.4f}{total_support:>10d}")
    lines.append(
        f"{'weighted avg':<16}{weighted_precision:>10.4f}{weighted_recall:>10.4f}{weighted_f1:>10.4f}{total_support:>10d}"
    )
    return "\n".join(lines) + "\n"


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


def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Optional[float]]:
    labels = sorted(np.unique(y_test).tolist())
    accuracy = _safe_div(float(np.sum(y_test == y_pred)), float(len(y_test)))

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    total_support = 0

    for label in labels:
        precision, recall, f1, support = _per_class_metrics(y_test, y_pred, int(label))
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        total_support += support

    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy),
        "precision": float(_safe_div(weighted_precision, float(total_support))),
        "recall": float(_safe_div(weighted_recall, float(total_support))),
        "f1_score": float(_safe_div(weighted_f1, float(total_support))),
        "roc_auc": None,
    }

    if len(labels) == 2:
        positive_class = int(labels[1])
        if 0 <= positive_class < y_proba.shape[1]:
            y_binary = (y_test == positive_class).astype(np.int64)
            metrics["roc_auc"] = binary_roc_auc(y_binary, y_proba[:, positive_class])
    elif len(labels) > 2:
        weighted_auc = 0.0
        weighted_count = 0
        for label in labels:
            idx = int(label)
            if not (0 <= idx < y_proba.shape[1]):
                continue
            y_binary = (y_test == idx).astype(np.int64)
            support = int(np.sum(y_binary))
            auc = binary_roc_auc(y_binary, y_proba[:, idx])
            if auc is None or support <= 0:
                continue
            weighted_auc += auc * support
            weighted_count += support
        if weighted_count > 0:
            metrics["roc_auc"] = float(weighted_auc / weighted_count)

    return metrics


def iter_windows(start: int, end: int, window_size: int, overlap: float) -> Iterable[Tuple[int, int]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0.0, 1.0)")
    step = max(1, int(round(window_size * (1.0 - overlap))))
    cursor = start
    while cursor + window_size <= end:
        yield cursor, cursor + window_size
        cursor += step


def build_feature_table(
    loader: EEGDataLoader,
    extractor: EEGFeatureExtractor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[Dict[str, object]] = []
    errors: List[str] = []

    for participant_id in loader.participants:
        print(f"Processing {participant_id}...")
        try:
            payload = loader.load_json(participant_id)
            n_before = len(rows)
            ignored_keys: List[str] = []

            # Pop entries to reduce memory pressure while processing huge JSON files.
            for segment_key in list(payload.keys()):
                segment_data = payload.pop(segment_key)
                cycle_idx, phase_name, normalized_phase = loader.parse_segment_key(segment_key)
                if phase_name is None:
                    ignored_keys.append(segment_key)
                    continue

                segment_arr = np.asarray(segment_data, dtype=np.float32)
                if segment_arr.ndim != 2:
                    ignored_keys.append(segment_key)
                    continue

                n_channels, n_samples = segment_arr.shape
                if n_samples < WINDOW_SIZE:
                    continue

                for w_start, w_end in iter_windows(0, n_samples, WINDOW_SIZE, WINDOW_OVERLAP):
                    window = segment_arr[:, w_start:w_end]
                    features = extractor.extract_features(window)
                    row: Dict[str, object] = {
                        "participant_id": participant_id,
                        "cycle_index": cycle_idx if cycle_idx is not None else -1,
                        "segment_key": segment_key,
                        "phase_name": phase_name,
                        "phase_text": normalized_phase,
                        "phase_code": int(loader.phase_code_map[phase_name]),
                        "n_channels": int(n_channels),
                        "window_start_sample": int(w_start),
                        "window_end_sample": int(w_end),
                    }
                    row.update(features)
                    rows.append(row)

            n_after = len(rows)
            print(f"  Extracted {n_after - n_before} labeled windows.")
            if ignored_keys:
                print(f"  Ignored {len(ignored_keys)} unrecognized segments.")
        except Exception as exc:
            errors.append(f"{participant_id}: {exc}")
            print(f"  Skipped due to error: {exc}")

    if not rows:
        detail = "\n".join(errors) if errors else "No windows extracted."
        raise RuntimeError(f"Could not build dataset from sub_*.json files.\nDetails:\n{detail}")

    non_feature_keys = {
        "participant_id",
        "cycle_index",
        "segment_key",
        "phase_name",
        "phase_text",
        "phase_code",
        "n_channels",
        "window_start_sample",
        "window_end_sample",
    }
    feature_keys = [k for k in rows[0].keys() if k not in non_feature_keys]

    X = np.array([[float(r[k]) for k in feature_keys] for r in rows], dtype=np.float64)
    y = np.array([int(r["phase_code"]) for r in rows], dtype=np.int64)
    groups = np.array([str(r["participant_id"]) for r in rows], dtype=object)
    return X, y, groups


def group_train_test_split(groups: np.ndarray, test_size: float, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("Need at least two participants for subject-wise split.")

    rng = np.random.default_rng(random_seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_test_groups = int(round(len(unique_groups) * test_size))
    n_test_groups = min(max(1, n_test_groups), len(unique_groups) - 1)
    test_group_set = set(shuffled[:n_test_groups].tolist())

    test_mask = np.array([g in test_group_set for g in groups], dtype=bool)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Subject-wise split failed to create non-empty train/test sets.")

    return train_idx, test_idx


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
):
    label_to_index = {int(lbl): idx for idx, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for y_t, y_p in zip(y_true.tolist(), y_pred.tolist()):
        if int(y_t) in label_to_index and int(y_p) in label_to_index:
            cm[label_to_index[int(y_t)], label_to_index[int(y_p)]] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    return plt


def _binary_roc_curve_points(y_true_binary: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = y_true_binary.astype(np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    scores_sorted = scores[order]

    tp = 0
    fp = 0
    tpr_values = [0.0]
    fpr_values = [0.0]

    i = 0
    n = len(scores_sorted)
    while i < n:
        current_score = scores_sorted[i]
        while i < n and scores_sorted[i] == current_score:
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        tpr_values.append(_safe_div(float(tp), float(n_pos)))
        fpr_values.append(_safe_div(float(fp), float(n_neg)))

    if tpr_values[-1] != 1.0 or fpr_values[-1] != 1.0:
        tpr_values.append(1.0)
        fpr_values.append(1.0)

    return np.array(fpr_values), np.array(tpr_values)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"):
    fpr, tpr = _binary_roc_curve_points(y_true, y_proba)
    roc_auc = binary_roc_auc(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    if roc_auc is None:
        label = "ROC curve"
    else:
        label = f"ROC curve (AUC = {roc_auc:.2f})"
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=label)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt


def _create_run_output_dir(base_results_dir: str | Path, model_name: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_results_dir) / "baseline" / model_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_run_outputs(
    out_dir: Path,
    metrics: Dict[str, Optional[float]],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: List[int],
    class_names: List[str],
    config_snapshot: Dict[str, object],
) -> None:
    payload = {
        "metrics": metrics,
        "n_test": int(len(y_test)),
        "label_distribution_test": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "class_labels": class_labels,
        "class_names": class_names,
        "config": config_snapshot,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (out_dir / "classification_report.txt").write_text(
        classification_report_text(y_test, y_pred, labels=class_labels, class_names=class_names),
        encoding="utf-8",
    )

    if SAVE_PLOTS:
        cm_fig = plot_confusion_matrix(y_test, y_pred, class_labels, class_names, title="Confusion Matrix")
        cm_fig.savefig(out_dir / f"confusion_matrix.{PLOT_FORMAT}", dpi=PLOT_DPI)
        plt.close(cm_fig.gcf())

        if len(class_labels) == 2:
            positive_class = class_labels[1]
            if 0 <= int(positive_class) < y_proba.shape[1]:
                binary_true = (y_test == positive_class).astype(int)
                roc_fig = plot_roc_curve(binary_true, y_proba[:, int(positive_class)], title=f"ROC Curve ({class_names[1]})")
                roc_fig.savefig(out_dir / f"roc_curve.{PLOT_FORMAT}", dpi=PLOT_DPI)
                plt.close(roc_fig.gcf())


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Creativity Baseline (PyTorch + Subject-Wise Split)")
    print("=" * 60)

    set_random_seed(RANDOM_SEED)

    loader = EEGDataLoader(
        data_dir=DATA_DIR,
        participants=PARTICIPANTS,
        phase_code_map=PHASE_CODE_MAP,
        json_phase_to_canonical=JSON_PHASE_TO_CANONICAL,
    )
    extractor = EEGFeatureExtractor(sampling_rate=SAMPLING_RATE)

    X, y, groups = build_feature_table(loader, extractor)
    train_idx, test_idx = group_train_test_split(groups, test_size=TEST_SIZE, random_seed=RANDOM_SEED)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Train participants: {sorted(set(groups_train.tolist()))}")
    print(f"Test participants: {sorted(set(groups_test.tolist()))}")

    observed_labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
    inverse_phase_map = {v: k for k, v in PHASE_CODE_MAP.items()}
    class_names = [inverse_phase_map.get(lbl, str(lbl)) for lbl in observed_labels]

    requested_models = [m.strip().lower() for m in MODEL_VARIANTS if str(m).strip()]
    if not requested_models:
        requested_models = ["torch_linear", "torch_mlp"]

    input_dim = int(X_train.shape[1])
    num_classes = int(max(PHASE_CODE_MAP.values()) + 1)

    for model_name in requested_models:
        model = None
        model_config: Dict[str, object] = {}

        if model_name in {"torch_linear", "linear", "logreg", "logistic", "logistic_regression"}:
            model_name = "torch_linear"
            model = TorchEEGModel(
                input_dim=input_dim,
                num_classes=num_classes,
                variant="linear",
                model_config=TORCH_LINEAR_CONFIG,
            )
            model_config = dict(TORCH_LINEAR_CONFIG)
            print("\nTraining PyTorch linear baseline model...")
        elif model_name in {"torch_mlp", "mlp", "xgboost", "xgb"}:
            model_name = "torch_mlp"
            model = TorchEEGModel(
                input_dim=input_dim,
                num_classes=num_classes,
                variant="mlp",
                model_config=TORCH_MLP_CONFIG,
            )
            model_config = dict(TORCH_MLP_CONFIG)
            print("\nTraining PyTorch MLP baseline model...")
        else:
            print(f"\nSkipping unknown model variant: {model_name}")
            continue

        print(f"  Device: {model.device}")
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = model.evaluate(X_test, y_test, y_pred, y_proba)

        print("\n" + "=" * 60)
        print(f"MODEL EVALUATION RESULTS ({model_name})")
        print("=" * 60)
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\nDetailed Classification Report:")
        print(classification_report_text(y_test, y_pred, labels=observed_labels, class_names=class_names))

        out_dir = _create_run_output_dir(RESULTS_DIR, model_name)
        config_snapshot = {
            "MODEL_NAME": model_name,
            "MODEL_VARIANTS": requested_models,
            "MODEL_CONFIG": model_config,
            "DATA_DIR": str(DATA_DIR),
            "PARTICIPANTS": list(PARTICIPANTS),
            "SAMPLING_RATE": int(SAMPLING_RATE),
            "WINDOW_SIZE": int(WINDOW_SIZE),
            "WINDOW_OVERLAP": float(WINDOW_OVERLAP),
            "TEST_SIZE": float(TEST_SIZE),
            "RANDOM_SEED": int(RANDOM_SEED),
            "PHASE_CODE_MAP": PHASE_CODE_MAP,
            "JSON_PHASE_TO_CANONICAL": JSON_PHASE_TO_CANONICAL,
            "SAVE_PLOTS": bool(SAVE_PLOTS),
            "PLOT_FORMAT": str(PLOT_FORMAT),
            "PLOT_DPI": int(PLOT_DPI),
            "TORCH_VERSION": torch.__version__,
            "DEVICE": str(model.device),
        }
        _save_run_outputs(
            out_dir=out_dir,
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=observed_labels,
            class_names=class_names,
            config_snapshot=config_snapshot,
        )
        print(f"\nSaved outputs to: {out_dir}")

    print("\nComplete.")

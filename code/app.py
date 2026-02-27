# EEG baseline training from `sub_*.json` files.
# The JSON files already contain segmented EEG arrays with phase labels in keys.

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore")

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
    BASELINE_MODEL_CONFIG = dict(
        getattr(
            project_config,
            "BASELINE_MODEL_CONFIG",
            {
                "max_iter": 1000,
                "solver": "lbfgs",
                "C": 1.0,
                "random_state": RANDOM_SEED,
                "class_weight": "balanced",
            },
        )
    )
    XGBOOST_CONFIG = dict(
        getattr(
            project_config,
            "XGBOOST_CONFIG",
            {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": RANDOM_SEED,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
            },
        )
    )
    MODEL_VARIANTS = list(getattr(project_config, "MODEL_VARIANTS", ["logreg", "xgboost"]))
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
    BASELINE_MODEL_CONFIG = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "C": 1.0,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }
    XGBOOST_CONFIG = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": RANDOM_SEED,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
    }
    MODEL_VARIANTS = ["logreg", "xgboost"]
    PHASE_CODE_MAP = {"RST": 0, "IDG": 1, "IDE": 2, "IDR": 3}
    JSON_PHASE_TO_CANONICAL = {
        "rest": "RST",
        "idea generation": "IDG",
        "idea evolution": "IDE",
        "idea rating": "IDR",
    }


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


class BaselineEEGModel:
    """Logistic Regression baseline model."""

    def __init__(self, model_config: Optional[Dict[str, object]] = None):
        config = dict(model_config or {})
        config.setdefault("max_iter", 1000)
        config.setdefault("random_state", RANDOM_SEED)
        self.model = LogisticRegression(**config)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

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


class XGBoostEEGModel:
    """XGBoost baseline model for non-linear multiclass decision boundaries."""

    def __init__(self, model_config: Optional[Dict[str, object]] = None):
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it with: pip install xgboost")

        config = dict(model_config or {})
        config.setdefault("random_state", RANDOM_SEED)
        config.setdefault("objective", "multi:softprob")
        config.setdefault("eval_metric", "mlogloss")
        config.setdefault("num_class", len(PHASE_CODE_MAP))
        self.model = XGBClassifier(**config)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

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


def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
    }

    try:
        unique_count = len(np.unique(y_test))
        if unique_count == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        elif unique_count > 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
    except Exception:
        metrics["roc_auc"] = None

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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    class_names: List[str],
    title: str = "Confusion Matrix",
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
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


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
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
        classification_report(y_test, y_pred, labels=class_labels, target_names=class_names, zero_division=0),
        encoding="utf-8",
    )

    if SAVE_PLOTS:
        cm_fig = plot_confusion_matrix(y_test, y_pred, class_labels, class_names, title="Confusion Matrix")
        cm_fig.savefig(out_dir / f"confusion_matrix.{PLOT_FORMAT}", dpi=PLOT_DPI)
        plt.close(cm_fig.gcf())

        if len(class_labels) == 2:
            positive_class = class_labels[1]
            positive_col = class_labels.index(positive_class)
            binary_true = (y_test == positive_class).astype(int)
            roc_fig = plot_roc_curve(binary_true, y_proba[:, positive_col], title=f"ROC Curve ({class_names[1]})")
            roc_fig.savefig(out_dir / f"roc_curve.{PLOT_FORMAT}", dpi=PLOT_DPI)
            plt.close(roc_fig.gcf())


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Creativity Baseline (sub_*.json + Subject-Wise Split)")
    print("=" * 60)

    loader = EEGDataLoader(
        data_dir=DATA_DIR,
        participants=PARTICIPANTS,
        phase_code_map=PHASE_CODE_MAP,
        json_phase_to_canonical=JSON_PHASE_TO_CANONICAL,
    )
    extractor = EEGFeatureExtractor(sampling_rate=SAMPLING_RATE)

    X, y, groups = build_feature_table(loader, extractor)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise RuntimeError("Need at least two participants for subject-wise split.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

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
        requested_models = ["logreg", "xgboost"]

    for model_name in requested_models:
        model = None
        model_config: Dict[str, object] = {}

        if model_name in {"logreg", "logistic", "logistic_regression"}:
            model_name = "logreg"
            model = BaselineEEGModel(model_config=BASELINE_MODEL_CONFIG)
            model_config = dict(BASELINE_MODEL_CONFIG)
            print("\nTraining Logistic Regression baseline model...")
        elif model_name in {"xgboost", "xgb"}:
            model_name = "xgboost"
            if XGBClassifier is None:
                print("\nSkipping XGBoost: package not installed in current interpreter.")
                continue
            model = XGBoostEEGModel(model_config=XGBOOST_CONFIG)
            model_config = dict(XGBOOST_CONFIG)
            print("\nTraining XGBoost baseline model...")
        else:
            print(f"\nSkipping unknown model variant: {model_name}")
            continue

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
        print(classification_report(y_test, y_pred, labels=observed_labels, target_names=class_names, zero_division=0))

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

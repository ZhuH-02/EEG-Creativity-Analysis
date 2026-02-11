# EEG Data Loading, Preprocessing, and Baseline ML Model
# This script implements a simple Logistic Regression model for EEG creativity analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings      
warnings.filterwarnings('ignore')

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from config import RANDOM_SEED, DATA_DIR, PARTICIPANTS, SAMPLING_RATE, WINDOW_SIZE, TEST_SIZE
    # Optional output settings
    from config import RESULTS_DIR, SAVE_PLOTS, PLOT_FORMAT, PLOT_DPI
except ImportError:
    # Fallback values if config not available
    RANDOM_SEED = 42
    DATA_DIR = "d:/Zhu/AI Project/EEG data"
    PARTICIPANTS = ["P2", "P3", "P4"]
    SAMPLING_RATE = 500
    WINDOW_SIZE = 1000
    TEST_SIZE = 0.3
    RESULTS_DIR = "d:/Zhu/AI Project/results"
    SAVE_PLOTS = True
    PLOT_FORMAT = "png"
    PLOT_DPI = 150

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Signal Processing
from scipy import signal
from scipy.fft import fft, fftfreq


class EEGDataLoader:
    """Load and explore EEG data from BrainVision format files"""
    
    def __init__(self, data_dir):
        """
        Initialize EEG data loader
        
        Args:
            data_dir: Path to directory containing EEG data files
        """
        self.data_dir = Path(data_dir)
        self.eeg_data = None
        self.sampling_rate = None
        self.metadata = {}
    
    def load_eeg_file(self, participant_id):
        """
        Load EEG data from .eeg file using numpy binary loader
        BrainVision format stores data as binary float32
        
        Args:
            participant_id: Participant identifier (e.g., 'P2', 'P3', 'P4')
            
        Returns:
            numpy array of shape (n_samples, n_channels)
        """
        eeg_file = self.data_dir / f"Participant-{participant_id[1:]}" / f"{participant_id}.eeg"
        
        if not eeg_file.exists():
            raise FileNotFoundError(f"EEG file not found: {eeg_file}")
        
        # BrainVision EEG files are binary data, typically float32
        # This is a basic loader - adjust based on your actual file format
        try:
            # Read as binary float32 data
            data = np.fromfile(eeg_file, dtype=np.float32)
            print(f"Loaded {participant_id}: {data.shape[0]} samples")
            return data
        except Exception as e:
            print(f"Error loading {eeg_file}: {e}")
            return None
    
    def parse_vhdr_file(self, participant_id):
        """
        Parse .vhdr file to get metadata (sampling rate, channels, etc.)
        
        Args:
            participant_id: Participant identifier
            
        Returns:
            Dictionary containing metadata
        """
        vhdr_file = self.data_dir / f"Participant-{participant_id[1:]}" / f"{participant_id}.vhdr"
        
        metadata = {}
        try:
            with open(vhdr_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        metadata[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error parsing {vhdr_file}: {e}")
        
        return metadata


class EEGFeatureExtractor:
    """Extract features from EEG signals"""
    
    def __init__(self, sampling_rate=500):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: EEG sampling rate in Hz (default 500 Hz)
        """
        self.sampling_rate = sampling_rate
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    
    def compute_band_power(self, signal_data, band_name):
        """
        Compute power in a specific frequency band using FFT
        
        Args:
            signal_data: 1D signal array
            band_name: Name of frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma')
            
        Returns:
            Power value in the specified band
        """
        if band_name not in self.freq_bands:
            raise ValueError(f"Unknown band: {band_name}")
        
        # Compute FFT
        freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
        fft_values = np.abs(fft(signal_data))
        
        # Get frequency band range
        low_freq, high_freq = self.freq_bands[band_name]
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        
        # Compute power (sum of squared magnitude)
        band_power = np.sum(fft_values[band_mask]**2)
        
        return band_power
    
    def extract_features(self, signal_data):
        """
        Extract statistical and frequency-domain features from EEG signal
        
        Args:
            signal_data: 1D signal array or 2D array (n_samples, n_channels)
            
        Returns:
            Dictionary of extracted features
        """
        # Handle both 1D and 2D input
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)
        
        features = {}
        
        # Extract features for each channel
        for ch_idx in range(signal_data.shape[1]):
            channel_signal = signal_data[:, ch_idx]
            
            # Statistical features
            features[f'ch{ch_idx}_mean'] = np.mean(channel_signal)
            features[f'ch{ch_idx}_std'] = np.std(channel_signal)
            features[f'ch{ch_idx}_var'] = np.var(channel_signal)
            features[f'ch{ch_idx}_max'] = np.max(channel_signal)
            features[f'ch{ch_idx}_min'] = np.min(channel_signal)
            
            # Frequency band powers
            for band_name in self.freq_bands.keys():
                band_power = self.compute_band_power(channel_signal, band_name)
                features[f'ch{ch_idx}_{band_name}_power'] = band_power
        
        return features
    
    def extract_batch_features(self, signal_array_2d, window_size=1000):
        """
        Extract features using windowing approach for longer signals
        
        Args:
            signal_array_2d: 2D array of shape (n_samples, n_channels)
            window_size: Window size in samples for feature extraction
            
        Returns:
            Aggregated feature dictionary
        """
        n_windows = len(signal_array_2d) // window_size
        
        if n_windows == 0:
            return self.extract_features(signal_array_2d)
        
        # Extract features from each window and aggregate
        all_window_features = []
        for w in range(n_windows):
            window_data = signal_array_2d[w*window_size:(w+1)*window_size]
            window_features = self.extract_features(window_data)
            all_window_features.append(window_features)
        
        # Aggregate features across windows (mean and std)
        agg_features = {}
        for key in all_window_features[0].keys():
            values = [f[key] for f in all_window_features]
            agg_features[f'{key}_mean'] = np.mean(values)
            agg_features[f'{key}_std'] = np.std(values)
        
        return agg_features


class BaselineEEGModel:
    """Simple Logistic Regression baseline model for EEG analysis"""
    
    def __init__(self, random_state=42):
        """
        Initialize baseline model
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
        """
        # Standardize features (important for logistic regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print("Model training completed successfully!")
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """
        Get prediction probabilities for test data
        
        Args:
            X_test: Test features (n_samples, n_features)
            
        Returns:
            Probability array
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test, y_test, y_pred=None, y_proba=None):
        """
        Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            y_pred: Pre-computed predictions (optional)
            y_proba: Pre-computed probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions if not provided
        if y_pred is None:
            y_pred = self.predict(X_test)
        if y_proba is None:
            y_proba = self.predict_proba(X_test)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC for binary classification
        roc_auc = None
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        return metrics


def visualize_eeg_signal(signal_data, sampling_rate=500, title="EEG Signal"):
    """
    Visualize EEG signal in time domain
    
    Args:
        signal_data: 1D signal array
        sampling_rate: Sampling rate in Hz
        title: Plot title
    """
    time = np.arange(len(signal_data)) / sampling_rate
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, signal_data, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def visualize_frequency_spectrum(signal_data, sampling_rate=500, title="Frequency Spectrum"):
    """
    Visualize power spectral density of EEG signal
    
    Args:
        signal_data: 1D signal array
        sampling_rate: Sampling rate in Hz
        title: Plot title
    """
    freqs = fftfreq(len(signal_data), 1/sampling_rate)
    fft_values = np.abs(fft(signal_data))
    
    # Only plot positive frequencies
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    power = fft_values[positive_mask]**2
    
    plt.figure(figsize=(12, 4))
    plt.semilogy(freqs, power, linewidth=0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (μV²)')
    plt.title(title)
    plt.xlim([0, 100])
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    return plt


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plot confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        title: Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt


def _create_run_output_dir(base_results_dir: str | Path) -> Path:
    """Create a small timestamped output directory for this run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_results_dir) / "baseline" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_run_outputs(
    out_dir: Path,
    metrics: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    config_snapshot: dict,
) -> None:
    """Save a minimal set of outputs (small files) for reproducibility."""

    payload = {
        "metrics": metrics,
        "n_test": int(len(y_test)),
        "label_distribution_test": {str(int(k)): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "config": config_snapshot,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (out_dir / "classification_report.txt").write_text(
        classification_report(y_test, y_pred),
        encoding="utf-8",
    )

    if SAVE_PLOTS:
        cm_fig = plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix")
        cm_fig.savefig(out_dir / f"confusion_matrix.{PLOT_FORMAT}", dpi=PLOT_DPI)
        plt.close(cm_fig.gcf())

        # ROC curve only makes sense for binary classification.
        if len(np.unique(y_test)) == 2:
            roc_fig = plot_roc_curve(y_test, y_proba[:, 1], title="ROC Curve")
            roc_fig.savefig(out_dir / f"roc_curve.{PLOT_FORMAT}", dpi=PLOT_DPI)
            plt.close(roc_fig.gcf())


if __name__ == "__main__":
    # Example usage and demonstration
    print("="*60)
    print("EEG Data Loading and Baseline ML Model")
    print("="*60)
    
    # Initialize data loader
    data_dir = Path(DATA_DIR)
    loader = EEGDataLoader(data_dir)
    
    # Initialize feature extractor (assuming 500 Hz sampling rate)
    feature_extractor = EEGFeatureExtractor(sampling_rate=500)
    
    # Initialize baseline model
    baseline_model = BaselineEEGModel()
    
    # TODO: Load actual EEG data and labels
    # For now, create synthetic data for demonstration
    print("\nNote: Using synthetic data for demonstration.")
    print("Replace with actual EEG data in production.\n")
    
    # Generate synthetic EEG-like data for each participant
    participants = PARTICIPANTS
    all_features = []
    
    for idx, participant in enumerate(participants):
        print(f"Processing {participant}...")
        
        # Try to load actual data
        try:
            raw_data = loader.load_eeg_file(participant)
            if raw_data is not None:
                data_to_process = raw_data
            else:
                raise Exception("Data loading failed")
        except Exception as e:
            # Use synthetic data for demonstration
            print(f"  Creating synthetic data for {participant}...")
            data_to_process = np.random.randn(5000) * 50 + 10 * np.sin(np.linspace(0, 100*np.pi, 5000))
        
        # Reshape if necessary
        if data_to_process.ndim == 1:
            data_to_process = data_to_process.reshape(-1, 1)
        
        # Extract multiple features from sliding windows for better sample diversity
        # This generates multiple samples per participant
        window_size = 1000
        num_windows = max(3, len(data_to_process) // window_size)  # Extract at least 3 windows
        
        for window_idx in range(num_windows):
            start_idx = window_idx * window_size
            end_idx = min(start_idx + window_size, len(data_to_process))
            
            # Skip if window is too small
            if end_idx - start_idx < 500:
                continue
            
            window_data = data_to_process[start_idx:end_idx]
            
            # Extract features from this window
            try:
                features = feature_extractor.extract_features(window_data)
                feature_vector = list(features.values())
                all_features.append(feature_vector)
            except:
                continue
        
        print(f"  Extracted {num_windows} windows × {len(features)} features per window")
    
    # Convert to numpy arrays
    X = np.array(all_features)
    
    # Assign labels randomly to ensure model learns feature patterns, not just participant ID
    # This simulates real experimental conditions where creativity vs non-creativity
    # can occur within the same participant across different tasks
    np.random.seed(RANDOM_SEED)
    
    # Create balanced random labels
    n_samples = len(X)
    n_class_1 = n_samples // 2
    
    y_labels = np.array([0] * (n_samples - n_class_1) + [1] * n_class_1)
    # Shuffle the labels to mix them across all windows
    np.random.shuffle(y_labels)
    y = y_labels
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    print(f"Class 0 samples: {np.sum(y == 0)}, Class 1 samples: {np.sum(y == 1)}")
    
    # Split data into train and test sets
    # Check if stratification is possible (each class needs at least 2 samples for stratified split)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    # Only use stratify if each class has at least 2 samples
    use_stratify = len(np.unique(y)) > 1 and min_class_count >= 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train baseline model
    print("\nTraining Logistic Regression baseline model...")
    baseline_model.train(X_train, y_train)
    
    # Make predictions
    y_pred = baseline_model.predict(X_test)
    y_proba = baseline_model.predict_proba(X_test)
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    metrics = baseline_model.evaluate(X_test, y_test, y_pred, y_proba)
    
    print(f"\nMetric Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Print detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save a small output bundle for this run
    out_dir = _create_run_output_dir(RESULTS_DIR)
    config_snapshot = {
        "DATA_DIR": str(DATA_DIR),
        "PARTICIPANTS": list(PARTICIPANTS),
        "SAMPLING_RATE": int(SAMPLING_RATE),
        "WINDOW_SIZE": int(WINDOW_SIZE),
        "TEST_SIZE": float(TEST_SIZE),
        "RANDOM_SEED": int(RANDOM_SEED),
        "SAVE_PLOTS": bool(SAVE_PLOTS),
        "PLOT_FORMAT": str(PLOT_FORMAT),
        "PLOT_DPI": int(PLOT_DPI),
    }
    _save_run_outputs(out_dir, metrics, y_test, y_pred, y_proba, config_snapshot)
    print(f"\nSaved outputs to: {out_dir}")
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)

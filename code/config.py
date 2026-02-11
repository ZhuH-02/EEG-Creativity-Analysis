# Configuration file for EEG Creativity Analysis Project
# Modify these settings based on your data characteristics

from pathlib import Path

# Project root directory (one level above /code)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Path to EEG data directory
DATA_DIR = str(PROJECT_ROOT / "EEG data")

# Participants to analyze (modify based on available data)
PARTICIPANTS = ["P2", "P3", "P4"]

# EEG file extension
EEG_FILE_EXTENSION = ".eeg"
VHDR_FILE_EXTENSION = ".vhdr"
VMRK_FILE_EXTENSION = ".vmrk"

# ============================================================================
# SIGNAL PROCESSING CONFIGURATION
# ============================================================================

# Sampling rate in Hz (IMPORTANT: Adjust to match your actual data)
# Common values: 250 Hz, 500 Hz, 1000 Hz
SAMPLING_RATE = 500

# Number of EEG channels (adjust if different)
# Common values: 8, 16, 32, 64, 128 channels
NUM_CHANNELS = 1  # Update based on your actual number of channels

# Signal data type (from .vhdr file)
# Common values: "float32", "int16", "int32"
DATA_TYPE = "float32"

# ============================================================================
# FREQUENCY BANDS (Hz)
# ============================================================================
# These are standard EEG frequency bands - modify if needed for your analysis

FREQUENCY_BANDS = {
    'delta': (0.5, 4),      # Delta: Deep sleep, unconscious
    'theta': (4, 8),        # Theta: Drowsiness, meditation
    'alpha': (8, 13),       # Alpha: Relaxation, creativity
    'beta': (13, 30),       # Beta: Active thinking, focus
    'gamma': (30, 100),     # Gamma: High-level processing
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# Bandpass filter range (Hz)
# Set to None to skip filtering, or specify range (low_freq, high_freq)
BANDPASS_FILTER = (0.5, 100)

# Notch filter frequency for power line noise removal (Hz)
# Set to None to skip, or 50 (EU) / 60 (US/Asia)
NOTCH_FILTER_FREQ = 50

# Artifact detection threshold (standard deviations)
# Samples exceeding this threshold will be marked as artifacts
ARTIFACT_THRESHOLD = 5.0

# ============================================================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================================================

# Window size for feature extraction (in samples)
# Real-world formula: window_size = SAMPLING_RATE * duration_in_seconds
# Example: 500 Hz * 2 seconds = 1000 samples
WINDOW_SIZE = 1000

# Window overlap (0.0 to 1.0, where 0.5 = 50% overlap)
WINDOW_OVERLAP = 0.5

# Welch's PSD parameters for frequency analysis
# Length of FFT windows
PSD_WINDOW_LENGTH = 256

# Overlap between PSD windows (in samples)
PSD_WINDOW_OVERLAP = 128

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Train-test split ratio
TEST_SIZE = 0.3

# Validation split ratio (for cross-validation)
VALIDATION_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# MODEL CONFIGURATION - WEEK 1 (Baseline)
# ============================================================================

# Logistic Regression parameters
BASELINE_MODEL_CONFIG = {
    'max_iter': 1000,
    'solver': 'lbfgs',      # 'lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'
    'C': 1.0,               # Regularization strength (lower = stronger regularization)
    'random_state': RANDOM_SEED,
    'class_weight': 'balanced',  # Handle class imbalance
}

# Whether to standardize features before training
STANDARDIZE_FEATURES = True

# ============================================================================
# MODEL CONFIGURATION - WEEK 2+ (Advanced Models)
# ============================================================================

# Random Forest parameters
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
}

# Support Vector Machine parameters
SVM_CONFIG = {
    'kernel': 'rbf',        # 'linear', 'poly', 'rbf', 'sigmoid'
    'C': 1.0,
    'gamma': 'scale',
    'random_state': RANDOM_SEED,
}

# XGBoost parameters
XGBOOST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': RANDOM_SEED,
}

# ============================================================================
# DEEP LEARNING CONFIGURATION - WEEK 3+
# ============================================================================

# CNN parameters
CNN_CONFIG = {
    'input_shape': None,           # Will be determined by data
    'kernel_size': 5,
    'num_filters': [32, 64, 128],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
}

# LSTM parameters
LSTM_CONFIG = {
    'input_shape': None,           # Will be determined by data
    'units': [64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'bidirectional': True,
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Directory for saving results and plots
RESULTS_DIR = str(PROJECT_ROOT / "results")

# Directory for saving trained models
MODELS_DIR = str(PROJECT_ROOT / "models")

# Whether to save plots to file
SAVE_PLOTS = True

# Plot format: 'png', 'pdf', 'svg', etc.
PLOT_FORMAT = 'png'

# Plot DPI (resolution)
PLOT_DPI = 150

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# Log file path
LOG_FILE = str(PROJECT_ROOT / "experiment.log")

# ============================================================================
# CROSS-VALIDATION CONFIGURATION
# ============================================================================

# Number of folds for K-fold cross-validation
NUM_FOLDS = 5

# Whether to perform stratified k-fold (recommended for imbalanced data)
STRATIFIED_KFOLD = True

# ============================================================================
# INTERPRETABILITY CONFIGURATION (Week 4+)
# ============================================================================

# Number of top features to display in importance plots
TOP_N_FEATURES = 15

# Whether to compute SHAP values (computationally expensive)
COMPUTE_SHAP = True

# ============================================================================
# NOTES & INSTRUCTIONS
# ============================================================================
"""
QUICK START:
1. Verify SAMPLING_RATE matches your actual EEG data
2. Update DATA_DIR if EEG data is in a different location
3. Confirm PARTICIPANTS list matches your data files
4. Check NUM_CHANNELS matches your EEG system

TROUBLESHOOTING:
- If you get "Data loading failed", check the file paths in DATA_DIR
- If models perform poorly, check if SAMPLING_RATE is correct
- If processing is slow, increase WINDOW_SIZE or reduce NUM_CHANNELS

CUSTOMIZATION:
- Modify FREQUENCY_BANDS if studying specific brain rhythms
- Adjust WINDOW_SIZE based on your analysis window duration
- Tune model parameters in MODEL_CONFIG dicts based on results
- Change TEST_SIZE for different train-test splits

For more information, see README.md and 5_WEEK_DEVELOPMENT_PLAN.md
"""

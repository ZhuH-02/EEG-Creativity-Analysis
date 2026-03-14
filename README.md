# EEG Creativity Analysis

Reproducible EEG phase-classification pipeline in Python and PyTorch for:

- EEG data auditing and feature extraction
- Subject-wise training, validation, and test evaluation
- Inference efficiency profiling
- Failure analysis, robustness testing, monitoring simulation, and adaptation experiments

## Overview

The project uses participant JSON files (`sub_XX.json`) as the main source for segmented EEG windows. From those windows, the code extracts a compact feature set with time-domain and frequency-band statistics, standardizes the features, and trains PyTorch classifiers to predict creativity-task phases.

The current codebase supports two main workflows:

- `code/app.py`: baseline subject-wise train/test experiments
- `code/train_milestone2.py`: expanded training pipeline with train/val/test split, early stopping, richer metrics, saved checkpoints, and profiling

On top of that, the repository now includes separate CLIs for:

- anticipated failure checks and stress tests
- robustness, calibration, and FGSM evaluation
- offline monitoring simulation
- drift adaptation experiments

## Dataset

Source DOI:

`10.17632/24yp3xp58b.1`

Expected local layout:

```text
EEG data/
  Participant-2/
    sub_02.json
    P2.eeg
    P2.vhdr
    P2.vmrk
  Participant-3/
    sub_03.json
    P3.eeg
  ...
```

File usage:

- `sub_XX.json`: primary training and evaluation source
- `P*.eeg`: raw EEG stream used by the audit workflow
- `P*.vhdr`, `P*.vmrk`: optional metadata files

## Label Mapping

Canonical labels used across training and evaluation:

- `RST` -> `0` (Rest)
- `IDG` -> `1` (Idea Generation)
- `IDE` -> `2` (Idea Evolution)
- `IDR` -> `3` (Idea Rating)

## Repository Structure

```text
.
├── code/
│   ├── adaptation_eval.py
│   ├── analysis_utils.py
│   ├── app.py
│   ├── config.py
│   ├── failure_checks.py
│   ├── monitoring_sim.py
│   ├── phase1_data_selection_audit.py
│   ├── report_plots.py
│   ├── robustness_eval.py
│   └── train_milestone2.py
├── notebooks/
├── outputs/
│   └── phase1_data_selection/
├── results/
│   └── baseline/
├── runs/
│   ├── <training runs>/
│   └── final_eval/
├── requirements.txt
└── EEG data/
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If VS Code cannot import project modules, ensure the interpreter is:

`<workspace>\.venv\Scripts\python.exe`

## Configuration

Main project settings live in `code/config.py`.

Important settings:

- `DATA_DIR`, `PARTICIPANTS`
- `SAMPLING_RATE`, `WINDOW_SIZE`, `WINDOW_OVERLAP`
- `RANDOM_SEED`
- `PHASE_CODE_MAP`, `JSON_PHASE_TO_CANONICAL`
- `TORCH_LINEAR_CONFIG`, `TORCH_MLP_CONFIG`
- `RESULTS_DIR`

Training runs also persist their resolved configuration in each run folder as `config.json`.

## Main Workflows

### 1. Phase 1 Audit

```powershell
python code/phase1_data_selection_audit.py
```

Outputs:

- `outputs/phase1_data_selection/features.csv`
- `outputs/phase1_data_selection/file_audit.json`
- `outputs/phase1_data_selection/missingness.csv`
- `outputs/phase1_data_selection/summary_stats.csv`
- `outputs/phase1_data_selection/duplicates.csv`
- `outputs/phase1_data_selection/outliers_summary.csv`
- `outputs/phase1_data_selection/example_rows.csv`
- `outputs/phase1_data_selection/run_metadata.json`
- `outputs/phase1_data_selection/plot_histograms.png`
- `outputs/phase1_data_selection/plot_corr_heatmap.png`
- `outputs/phase1_data_selection/plot_windows_per_participant.png`

### 2. Baseline Training

```powershell
python code/app.py
```

This path:

- loads configured participant JSON files
- extracts EEG window features
- performs subject-wise train/test split
- trains PyTorch linear and MLP baselines
- saves baseline metrics and plots under `results/baseline/`

### 3. Expanded Train/Val/Test Training

```powershell
python code/train_milestone2.py --model torch_mlp --tag expanded_baseline_mlp
```

This path adds:

- participant-wise train/val/test split
- early stopping on validation macro-F1
- richer multiclass metrics
- saved best and last checkpoints
- learning curves
- inference efficiency profiling

Training artifacts are written to:

`runs/<timestamp>_<model>_<tag>/`

Expected files:

- `best_model.pt`
- `last_model.pt`
- `config.json`
- `metrics.json`
- `efficiency.json`
- `split_subjects.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `learning_curves.csv`
- `learning_curves.png`

## Final Evaluation Workflows

All final evaluation scripts load an existing training run directory and write outputs under:

`runs/final_eval/<timestamp>_<run_name>/`

Example source model directory:

`runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`

### Failure Checks and Stress Tests

```powershell
python code/failure_checks.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name failure_checks
```

Outputs:

- `stress_test_metrics.csv`
- `failure_catalog.json`
- `failure_catalog.md`
- `failure_summary.json`
- `failure_examples.csv` when flags are found
- `clean_classification_report.txt`

### Robustness, Calibration, and FGSM

```powershell
python code/robustness_eval.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name robustness_eval
```

Outputs:

- `robustness_metrics.csv`
- `fgsm_metrics.csv`
- `calibration_metrics.json`
- `reliability_bins.csv`
- `robustness_curve.png`
- `reliability_diagram.png`
- `confidence_histogram_clean.png`
- `confidence_histogram_adv.png`
- `clean_vs_perturbed_efficiency.csv`

### Monitoring Simulation

```powershell
python code/monitoring_sim.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name monitoring_sim
```

Outputs:

- `monitoring_log.csv`
- `drift_metrics.csv`
- `alerts.json`
- `monitoring_dashboard.png`

### Adaptation Experiment

```powershell
python code/adaptation_eval.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name adaptation_eval
```

Outputs:

- `adaptation_before_after.csv`
- `adaptation_metrics.json`
- `adaptation_efficiency.json`
- `resolved_failure_examples.csv`
- `adapted_model.pt`

## Shared Utilities

These scripts are internal support modules used by the CLIs above:

- `code/analysis_utils.py`: artifact loading, split reconstruction, evaluation helpers, profiling helpers
- `code/report_plots.py`: plotting helpers for calibration, robustness, and monitoring outputs

## Current Verified Outputs

The current repository contains successful smoke-run outputs under:

- `runs/final_eval/20260314_104325_smoke_failure/`
- `runs/final_eval/20260314_104706_smoke_robustness/`
- `runs/final_eval/20260314_105302_smoke_monitoring/`
- `runs/final_eval/20260314_105859_smoke_adaptation/`

These were executed against:

- `runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`

## Notes

- `app.py` keeps legacy alias handling for compatibility.
- The final-evaluation scripts reuse the saved checkpoint format from `train_milestone2.py`.
- Raw EEG data is intentionally not committed to Git.

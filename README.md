# EEG Creativity Analysis

Reproducible EEG phase-classification pipeline in Python and PyTorch for:

- EEG data auditing and feature extraction
- subject-wise training, validation, and test evaluation
- failure analysis, robustness, monitoring, and adaptation
- Milestone 4 continual learning, HITL, active learning, and end-to-end rollout

## Overview

The project uses participant JSON files (`sub_XX.json`) as the main segmented EEG source. From those windows, the code extracts compact time-domain and frequency-band features, standardizes them, and trains PyTorch classifiers to predict creativity-task phases.

The repository now covers the full milestone sequence:

- `code/phase1_data_selection_audit.py`: audit and feature extraction from raw EEG artifacts
- `code/train_pipeline.py`: Milestone 2 train/val/test training pipeline
- `code/failure_checks.py`, `code/robustness_eval.py`, `code/monitoring_sim.py`, `code/adaptation_eval.py`: Milestone 3 evaluation suite
- `code/continual_learning.py`, `code/hitl_active_learning.py`, `code/active_learning_eval.py`, `code/milestone4_pipeline.py`, `code/milestone4_report.py`, `code/demo_end_to_end.py`: Milestone 4 system components

## Dataset

Source DOI:

`10.17632/24yp3xp58b.1`

The project uses two related representations of the EEG study data:

- `sub_XX.json`: segmented EEG arrays used by the training and evaluation pipelines
- `P*.eeg`: raw continuous EEG files used by the Phase 1 audit workflow
- `P*.vhdr`, `P*.vmrk`: optional BrainVision metadata files when available

Current local copy used in this workspace:

- `EEG data/` contains `27` participant folders (`Participant-2` to `Participant-28`)
- each participant folder currently contains one `sub_XX.json` file and one `P*.eeg` file
- total local dataset footprint is about `34.4 GB`
- the current raw-data copy does not include `.vhdr` or `.vmrk`, so `code/phase1_data_selection_audit.py` uses its binary fallback layout inference for `.eeg` files

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

How the code uses these files:

- `code/app.py` and `code/train_pipeline.py` load `sub_XX.json` as the main supervised-learning source
- `code/phase1_data_selection_audit.py` reads `P*.eeg` for raw-data auditing, and uses `P*.vhdr` / `P*.vmrk` only if they are present
- all train/validation/test splits are participant-wise, so windows from the same participant do not leak across splits

### JSON Training Data

The training code expects each `sub_XX.json` file to store segmented EEG arrays keyed by task-phase names. The loader in `code/app.py` is written for keys shaped like `cycle_phase`, for example:

- `1_rest`
- `1_idea generation`
- `2_idea evolution`
- `2_idea rating`

Those segment names are normalized into the four canonical classes listed below. Each segment is then cut into fixed windows using the current project defaults from `code/config.py`:

- sampling rate: `500 Hz`
- window size: `1000` samples (`2` seconds)
- window overlap: `0.5`

For each window, the project extracts compact channel-averaged features:

- time-domain statistics: `mean`, `std`, `var`, `min`, `max`, `rms`, `skew`, `kurtosis`
- frequency features: `delta`, `theta`, `alpha`, `beta`, `gamma` bandpower
- simple ratios such as `alpha/beta` and `theta/alpha`

### Audit-Derived Tabular Data

`outputs/phase1_data_selection/features.csv` is a derived feature table produced by the audit script, not a raw source file. The current audit artifacts in this repo cover:

- `27` participants
- `300` windows per participant
- `8100` total windows

Key audit outputs include:

- `features.csv`: one row per extracted window
- `file_audit.json`: file presence checks plus inferred raw EEG layout details
- `summary_stats.csv`, `missingness.csv`, `duplicates.csv`, `outliers_summary.csv`: data quality summaries
- `example_rows.csv`: sample rows from the derived feature table

## Label Mapping

- `RST` -> `0` (Rest)
- `IDG` -> `1` (Idea Generation)
- `IDE` -> `2` (Idea Evolution)
- `IDR` -> `3` (Idea Rating)

## Repository Structure

```text
.
├── code/
│   ├── active_learning_eval.py
│   ├── adaptation_eval.py
│   ├── analysis_utils.py
│   ├── app.py
│   ├── config.py
│   ├── continual_learning.py
│   ├── demo_end_to_end.py
│   ├── failure_checks.py
│   ├── hitl_active_learning.py
│   ├── milestone4_pipeline.py
│   ├── milestone4_report.py
│   ├── monitoring_sim.py
│   ├── phase1_data_selection_audit.py
│   ├── report_plots.py
│   ├── robustness_eval.py
│   └── train_pipeline.py
├── notebooks/
├── outputs/
├── results/
├── runs/
├── EEG data/
├── MILESTONE4_SUBMISSION.md
└── README.md
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

## Core Workflows

### 1. Phase 1 Audit

```powershell
python code/phase1_data_selection_audit.py
```

Main outputs:

- `outputs/phase1_data_selection/features.csv`
- `outputs/phase1_data_selection/file_audit.json`
- `outputs/phase1_data_selection/missingness.csv`
- `outputs/phase1_data_selection/summary_stats.csv`
- `outputs/phase1_data_selection/duplicates.csv`
- `outputs/phase1_data_selection/outliers_summary.csv`
- `outputs/phase1_data_selection/example_rows.csv`
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
python code/train_pipeline.py --model torch_mlp --tag expanded_baseline_mlp
```

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

## Milestone 3 Evaluation

All Milestone 3 scripts load an existing training run and write outputs under `runs/final_eval/<timestamp>_<run_name>/`.

Verified source model:

`runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`

### Failure Checks

```powershell
python code/failure_checks.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name failure_checks
```

### Robustness and Calibration

```powershell
python code/robustness_eval.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name robustness_eval
```

### Monitoring

```powershell
python code/monitoring_sim.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name monitoring_sim
```

### Adaptation

```powershell
python code/adaptation_eval.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/final_eval --run_name adaptation_eval
```

## Milestone 4 Workflows

Milestone 4 artifacts are written under:

`runs/milestone4/<timestamp>_<run_name>/`

### Phase 2: Continual Learning

```powershell
python code/continual_learning.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/milestone4 --run_name phase2_acceptance
```

Main outputs:

- `continual_summary.json`
- `batch_metrics.csv`
- `update_history.csv`
- `update_decisions.csv`
- `buffer_stats.csv`
- `model_versions.csv`

### Phase 3: HITL Triggering

```powershell
python code/hitl_active_learning.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/milestone4 --run_name phase3_acceptance
```

Main outputs:

- `review_summary.json`
- `hitl_candidates.csv`
- `human_feedback_log.csv`
- `intervention_log.csv`

### Phase 4: Active Learning

```powershell
python code/active_learning_eval.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/milestone4 --run_name phase4_acceptance
```

Main outputs:

- `active_learning_summary.json`
- `strategy_summary.csv`
- `cycle_metrics.csv`
- `query_log.csv`
- `labeling_efficiency.csv`
- `active_learning_curve.png`

### Phase 5: End-to-End System

```powershell
python code/milestone4_pipeline.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --output_dir runs/milestone4 --run_name phase5_acceptance
```

Main outputs:

- `end_to_end_summary.json`
- `system_run_log.csv`
- `trigger_log.csv`
- `query_feedback_log.csv`
- `rollout_history.csv`
- `model_versions.csv`
- `system_timeline.png`

### Phase 6: Final Report

```powershell
python code/milestone4_report.py `
  --training_run_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp `
  --phase2_dir runs/milestone4/20260406_152452_phase2_acceptance `
  --phase3_dir runs/milestone4/20260406_153744_phase3_acceptance `
  --phase4_dir runs/milestone4/20260406_155249_phase4_acceptance `
  --phase5_dir runs/milestone4/20260406_161825_phase5_acceptance `
  --output_dir runs/milestone4 `
  --run_name phase6_acceptance
```

Main outputs:

- `project_summary_table.csv`
- `final_system_summary.json`
- `artifact_manifest.json`
- `system_comparison.png`
- `submission_notes.md`

### One-Command Demo

```powershell
python code/demo_end_to_end.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --device cpu --with_report
```

This wrapper:

- runs a smaller end-to-end Milestone 4 pipeline
- saves a fresh demo output directory under `runs/milestone4/`
- optionally generates a final report using the latest phase outputs

## Shared Utilities

- `code/analysis_utils.py`: artifact loading, split reconstruction, evaluation helpers, profiling helpers
- `code/report_plots.py`: plotting helpers for calibration, robustness, monitoring, and summary figures

## Verified Milestone 4 Outputs

Current verified acceptance runs:

- `runs/milestone4/20260406_152452_phase2_acceptance/`
- `runs/milestone4/20260406_153744_phase3_acceptance/`
- `runs/milestone4/20260406_155249_phase4_acceptance/`
- `runs/milestone4/20260406_161825_phase5_acceptance/`

Use `MILESTONE4_SUBMISSION.md` for the recommended demo order and submission checklist.

## Notes

- `app.py` keeps legacy alias handling for compatibility.
- historical training configs may reference old absolute data paths, and `analysis_utils.py` now falls back to the local `EEG data/` directory when needed.
- raw EEG data is intentionally not committed to Git.

# Milestone-2 Training Guide

This guide runs the new Milestone-2 entrypoint:

- `code/train_milestone2.py`
- participant-wise `train/val/test` split (no subject leakage)
- early stopping on validation macro-F1
- learning curves (`.csv` + `.png`)
- expanded metrics (macro/micro/weighted PRF1 + OvR AUROC + OvR PR-AUC)
- efficiency profiling (`efficiency.json`)

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Baseline Commands

### 1) Baseline Linear

```powershell
python code/train_milestone2.py --model torch_linear --tag baseline_linear
```

### 2) Baseline MLP

```powershell
python code/train_milestone2.py --model torch_mlp --hidden_layers 64,32 --dropout 0.2 --tag baseline_mlp
```

## Ablation Commands

### 3) Ablation: class weights OFF

```powershell
python code/train_milestone2.py --model torch_linear --no-use_class_weights --tag ablation_no_weights
```

### 4) Ablation: frequency features OFF (time-domain only)

```powershell
python code/train_milestone2.py --model torch_linear --no-use_freq_features --tag ablation_no_freq
```

## Profiling Command

### 5) Profiling-focused run (keeps profiling enabled explicitly)

```powershell
python code/train_milestone2.py --model torch_mlp --profile --inference_batch_size 256 --tag profiling
```

## Optional Common Overrides

```powershell
python code/train_milestone2.py `
  --model torch_mlp `
  --epochs 120 `
  --patience 15 `
  --batch_size 256 `
  --lr 1e-3 `
  --weight_decay 1e-4 `
  --seed 42 `
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 `
  --tag custom_run
```

## Output Structure

Each run writes:

`runs/<timestamp>_<model>_<tag>/`

Expected artifacts:

- `metrics.json`
- `efficiency.json`
- `learning_curves.csv`
- `learning_curves.png`
- `confusion_matrix.png`
- `classification_report.txt`
- `best_model.pt`
- `last_model.pt`
- `config.json`
- `split_subjects.json`

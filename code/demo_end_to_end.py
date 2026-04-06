from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command Milestone 4 demo wrapper for the end-to-end pipeline and final report."
    )
    parser.add_argument("--model_dir", type=str, default="runs/20260314_095455_torch_mlp_expanded_baseline_mlp")
    parser.add_argument("--output_dir", type=str, default="runs/milestone4")
    parser.add_argument("--run_name", type=str, default="milestone4_demo")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--with_report", action="store_true", help="Generate the Phase 6 report after the demo pipeline finishes.")
    return parser.parse_args()


def _latest_demo_dir(base_dir: Path, run_name: str, marker_file: str) -> Path:
    suffix = f"_{run_name}"
    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.endswith(suffix) and (path / marker_file).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find a demo output directory under {base_dir} for run name '{run_name}'.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_command(command: List[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_dir).resolve()

    pipeline_command = [
        sys.executable,
        str((repo_root / "code" / "milestone4_pipeline.py").resolve()),
        "--model_dir",
        args.model_dir,
        "--output_dir",
        args.output_dir,
        "--run_name",
        args.run_name,
        "--device",
        args.device,
        "--query_budget_per_trigger",
        "8",
        "--min_reviewed_for_update",
        "24",
        "--max_updates",
        "2",
        "--adaptation_epochs",
        "4",
        "--adaptation_patience",
        "2",
    ]
    _run_command(pipeline_command)
    demo_dir = _latest_demo_dir(output_root, args.run_name, "end_to_end_summary.json")

    report_dir = None
    if args.with_report:
        report_command = [
            sys.executable,
            str((repo_root / "code" / "milestone4_report.py").resolve()),
            "--training_run_dir",
            args.model_dir,
            "--phase5_dir",
            str(demo_dir),
            "--output_dir",
            args.output_dir,
            "--run_name",
            f"{args.run_name}_report",
        ]
        _run_command(report_command)
        report_dir = _latest_demo_dir(output_root, f"{args.run_name}_report", "final_system_summary.json")

    result: Dict[str, str] = {"demo_output_dir": str(demo_dir)}
    if report_dir is not None:
        result["report_output_dir"] = str(report_dir)
    print(result)


if __name__ == "__main__":
    main()

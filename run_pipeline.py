"""Run the complete MixedDataAnalysis pipeline in numerical order."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PIPELINE_STEPS = {
    1: "01_prepare_data.py",
    2: "02_descriptive_analysis.py",
    3: "03_prepare_model_data.py",
    4: "04_train_arima_arimax.py",
    5: "05_train_tree_models.py",
    6: "06_train_recurrent_models.py",
    7: "07_compare_models.py",
    8: "08_final_results.py",
}


def resolve_steps(args: argparse.Namespace) -> list[int]:
    steps = sorted(set(args.steps)) if args.steps else list(range(args.from_step, args.to_step + 1))
    invalid = [s for s in steps if s not in PIPELINE_STEPS]
    if invalid:
        raise ValueError(f"Unsupported steps: {invalid}")
    return steps


def run_step(root: Path, step: int, args: argparse.Namespace) -> None:
    script = root / PIPELINE_STEPS[step]
    command = [sys.executable, str(script)]
    if step == 4:
        command += ["--n-jobs", str(max(1, args.n_jobs))]
        if args.quick_step04:
            command.append("--quick")
    elif step == 5:
        command += ["--n-jobs", str(max(1, args.n_jobs))]
    elif step == 6:
        command += ["--n-jobs", str(max(1, args.recurrent_n_jobs)), "--stage2-top", str(args.recurrent_top)]
    elif step == 7:
        command += ["--bootstrap-reps", str(args.bootstrap_reps)]

    print("=" * 80, flush=True)
    print(f"Running Step {step:02d}: {script.name}", flush=True)
    print("=" * 80, flush=True)
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    completed = subprocess.run(command, cwd=root, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Step {step:02d} failed with exit code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-step", type=int, default=1)
    parser.add_argument("--to-step", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel workers for Steps 04 and 05.")
    parser.add_argument("--recurrent-n-jobs", type=int, default=1, help="Workers for Step 06. One is safest and deterministic.")
    parser.add_argument("--recurrent-top", type=int, default=2, help="Top recurrent representations entering architecture tuning.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--quick-step04", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    steps = resolve_steps(args)
    for step in steps:
        run_step(root, step, args)
    print(f"Pipeline completed successfully for steps: {steps}", flush=True)


if __name__ == "__main__":
    main()

"""
Isolated worker for Step 04 ARIMA/SARIMA and ARIMAX/SARIMAX-X evaluation.

The orchestrator launches this worker in a separate operating-system process for
rolling-validation configurations. Isolation allows a numerically unstable
statsmodels optimizer to be terminated without stopping the complete pipeline.
The worker writes one pickled result and exits immediately after the file is
flushed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path


def load_step_module(script_path: Path):
    """Load the Step 04 implementation while registering its module name."""
    specification = importlib.util.spec_from_file_location(
        "mixeddata_step04", script_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def parse_tuple(value: str) -> tuple[int, ...]:
    """Parse a comma-separated integer tuple from the command line."""
    return tuple(int(part) for part in value.split(","))


def main() -> None:
    """Evaluate one configuration and persist the complete summary/predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["arima", "arimax"], required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--seasonal-order", default="0,0,0,0")
    parser.add_argument("--target-days", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exogenous-set")
    parser.add_argument("--lag-pattern-json")
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    step_module = load_step_module(project_root / "04_train_arima_arimax.py")
    modeling, targets = step_module.load_inputs(
        project_root / "outputs/data/03_modeling_long_data.csv",
        project_root / "outputs/tables/03_next_day_target_trajectory.csv",
    )
    participant_series = step_module.build_participant_series(modeling)
    targets_by_key = step_module.target_lookup(targets)
    specification = step_module.StatisticalSpec(
        order=parse_tuple(args.order),
        seasonal_order=parse_tuple(args.seasonal_order),
    )
    target_days = parse_tuple(args.target_days)

    if args.kind == "arima":
        result = step_module.evaluate_arima_spec(
            specification,
            participant_series,
            targets_by_key,
            target_days,
            args.split,
        )
    else:
        if not args.exogenous_set or not args.lag_pattern_json or args.horizon is None:
            raise ValueError("ARIMAX worker arguments are incomplete.")
        lag_pattern = step_module.LagPattern(**json.loads(args.lag_pattern_json))
        result = step_module.evaluate_arimax_configuration(
            specification,
            args.exogenous_set,
            lag_pattern,
            args.horizon,
            participant_series,
            targets_by_key,
            target_days,
            args.split,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(("ok", result), output_file, protocol=pickle.HIGHEST_PROTOCOL)
        output_file.flush()
        os.fsync(output_file.fileno())


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # The orchestrator treats a missing/invalid result as a failed candidate.
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        os._exit(1)
    os._exit(0)

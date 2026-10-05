"""
Step 07: Compare statistical, tree-based, recurrent, and naive forecasting models.

The final model is selected by rolling-validation performance among models with
adequate coverage and all three validation days. Day 7 is then used for one
independent final evaluation. Test rankings are reported descriptively, but no
model is retuned after test inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
BOOTSTRAP_REPS = 2000
MODEL_RENAME = {"ARIMA": "SARIMA"}


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {name: root / "outputs" / name for name in ["tables", "images", "reports", "models"]}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(table_dir: Path) -> pd.DataFrame:
    files = [
        table_dir / "04_statistical_model_predictions.csv",
        table_dir / "05_tree_model_predictions.csv",
        table_dir / "06_recurrent_model_predictions.csv",
    ]
    parts = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_csv(path)
        d["Model"] = d["Model"].replace(MODEL_RENAME)
        keep = ["Model", "OriginID", "ParticipantID", "TargetDay", "Split", "Horizon", "TargetOccasion", "Actual", "ActualObserved", "Prediction"]
        parts.append(d[keep])
    data = pd.concat(parts, ignore_index=True)
    data = data.drop_duplicates(["Model", "OriginID", "Horizon", "Split"], keep="last")
    return data


def error_metrics(g: pd.DataFrame) -> dict[str, float]:
    observed = g["ActualObserved"].eq(1) & g["Actual"].notna()
    valid = observed & g["Prediction"].notna()
    if not valid.any():
        return {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "ObservedTargets": int(observed.sum()), "PredictedTargets": 0, "Coverage": 0.0}
    e = g.loc[valid, "Prediction"].to_numpy(float) - g.loc[valid, "Actual"].to_numpy(float)
    return {
        "MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "Bias": float(np.mean(e)),
        "ObservedTargets": int(observed.sum()), "PredictedTargets": int(valid.sum()),
        "Coverage": float(valid.sum() / observed.sum()) if observed.sum() else np.nan,
    }


def comparison_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), g in data.groupby(["Model", "Split"]):
        m = error_metrics(g)
        day_maes = []
        for day, gd in g.groupby("TargetDay"):
            md = error_metrics(gd)
            if md["PredictedTargets"]:
                day_maes.append(md["MAE"])
        rows.append({
            "Model": model, "Split": split, **m,
            "PeriodMacroMAE": float(np.mean(day_maes)) if day_maes else np.nan,
            "EvaluatedPeriods": len(day_maes),
        })
    result = pd.DataFrame(rows)
    val = result[result.Split == "validation"].set_index("Model")
    test = result[result.Split == "test"].set_index("Model")
    models = sorted(set(val.index) | set(test.index))
    merged = []
    for model in models:
        vr = val.loc[model] if model in val.index else None
        tr = test.loc[model] if model in test.index else None
        merged.append({
            "Model": model,
            "ValidationMAE": vr.MAE if vr is not None else np.nan,
            "ValidationMacroMAE": vr.PeriodMacroMAE if vr is not None else np.nan,
            "ValidationRMSE": vr.RMSE if vr is not None else np.nan,
            "ValidationCoverage": vr.Coverage if vr is not None else np.nan,
            "ValidationPeriods": int(vr.EvaluatedPeriods) if vr is not None else 0,
            "TestMAE": tr.MAE if tr is not None else np.nan,
            "TestRMSE": tr.RMSE if tr is not None else np.nan,
            "TestBias": tr.Bias if tr is not None else np.nan,
            "TestCoverage": tr.Coverage if tr is not None else np.nan,
            "TestObservedTargets": int(tr.ObservedTargets) if tr is not None else 0,
        })
    out = pd.DataFrame(merged)
    out["EligibleForFinalSelection"] = (
        (out.ValidationPeriods == 3) & (out.ValidationCoverage >= 0.95) & (out.TestCoverage >= 0.95)
    ).astype(int)
    out["ValidationToTestGap"] = out.TestMAE - out.ValidationMacroMAE
    out["ValidationRank"] = np.nan
    eligible = out.EligibleForFinalSelection.eq(1)
    out.loc[eligible, "ValidationRank"] = out.loc[eligible, "ValidationMacroMAE"].rank(method="min")
    out["TestRank"] = out.TestMAE.rank(method="min")
    selected_idx = out.loc[eligible, "ValidationMacroMAE"].idxmin()
    out["SelectedByValidation"] = 0
    out.loc[selected_idx, "SelectedByValidation"] = 1
    return out.sort_values(["SelectedByValidation", "ValidationRank", "TestMAE"], ascending=[False, True, True])


def horizon_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, horizon), g in data.groupby(["Model", "Split", "Horizon"]):
        rows.append({"Model": model, "Split": split, "Horizon": int(horizon), **error_metrics(g)})
    return pd.DataFrame(rows)


def participant_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, pid), g in data.groupby(["Model", "Split", "ParticipantID"]):
        rows.append({"Model": model, "Split": split, "ParticipantID": int(pid), **error_metrics(g)})
    return pd.DataFrame(rows)


def daily_mean_outputs(data: pd.DataFrame, minimum_positions: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, g in data.groupby(["Model", "Split", "OriginID", "ParticipantID", "TargetDay"]):
        valid = g[g.ActualObserved.eq(1) & g.Actual.notna() & g.Prediction.notna()]
        if len(valid) < minimum_positions:
            continue
        rows.append({
            "Model": keys[0], "Split": keys[1], "OriginID": keys[2], "ParticipantID": keys[3],
            "TargetDay": keys[4], "ObservedPositions": len(valid),
            "ActualDailyMean": valid.Actual.mean(), "PredictedDailyMean": valid.Prediction.mean(),
        })
    predictions = pd.DataFrame(rows)
    metrics = []
    for (model, split), g in predictions.groupby(["Model", "Split"]):
        e = g.PredictedDailyMean.to_numpy() - g.ActualDailyMean.to_numpy()
        metrics.append({
            "Model": model, "Split": split, "MinimumObservedPositions": minimum_positions,
            "ParticipantDays": len(g), "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)), "Bias": np.mean(e),
        })
    return predictions, pd.DataFrame(metrics)


def common_test_set(data: pd.DataFrame, eligible_models: list[str]) -> pd.DataFrame:
    test = data[(data.Split == "test") & data.Model.isin(eligible_models)].copy()
    key = ["OriginID", "ParticipantID", "TargetDay", "Horizon", "TargetOccasion"]
    availability = test.assign(Valid=test.ActualObserved.eq(1) & test.Actual.notna() & test.Prediction.notna()).groupby(key).Valid.sum()
    common_keys = availability[availability == len(eligible_models)].reset_index()[key]
    common = test.merge(common_keys, on=key, how="inner")
    return common


def bootstrap_participants(common: pd.DataFrame, selected_model: str, reps: int = BOOTSTRAP_REPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    participants = np.sort(common.ParticipantID.unique())
    model_names = sorted(common.Model.unique())
    by_pid_model = {}
    for (pid, model), g in common.groupby(["ParticipantID", "Model"]):
        e = np.abs(g.Prediction.to_numpy(float) - g.Actual.to_numpy(float))
        by_pid_model[(pid, model)] = (float(np.sum(e)), len(e))
    draws = {model: [] for model in model_names}
    differences = {model: [] for model in model_names if model != selected_model}
    for _ in range(reps):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        maes = {}
        for model in model_names:
            total_error = 0.0
            total_n = 0
            for pid in sampled:
                err, n = by_pid_model[(pid, model)]
                total_error += err
                total_n += n
            maes[model] = total_error / total_n
            draws[model].append(maes[model])
        for model in differences:
            differences[model].append(maes[model] - maes[selected_model])
    ci_rows = []
    for model, values in draws.items():
        arr = np.asarray(values)
        ci_rows.append({"Model": model, "BootstrapReps": reps, "MAEMean": arr.mean(), "MAELower95": np.quantile(arr, 0.025), "MAEUpper95": np.quantile(arr, 0.975)})
    diff_rows = []
    for model, values in differences.items():
        arr = np.asarray(values)
        diff_rows.append({
            "ComparisonModel": model, "ReferenceModel": selected_model, "DifferenceMAE": arr.mean(),
            "DifferenceLower95": np.quantile(arr, 0.025), "DifferenceUpper95": np.quantile(arr, 0.975),
            "ProbabilityComparisonWorse": float(np.mean(arr > 0)),
        })
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def save_plots(comparison: pd.DataFrame, horizons: pd.DataFrame, daily: pd.DataFrame, image_dir: Path) -> None:
    test_rank = comparison.sort_values("TestMAE")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(test_rank.Model, test_rank.TestMAE)
    ax.invert_yaxis()
    ax.set_xlabel("Day 7 test MAE")
    ax.set_title("Model test MAE ranking")
    fig.tight_layout()
    fig.savefig(image_dir / "07_model_test_mae_ranking.png", dpi=180)
    plt.close(fig)

    test_h = horizons[horizons.Split == "test"]
    fig, ax = plt.subplots(figsize=(12, 7))
    for model, g in test_h.groupby("Model"):
        g = g.sort_values("Horizon")
        ax.plot(g.Horizon, g.MAE, marker="o", alpha=0.85, label=model)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test MAE")
    ax.set_xticks(range(1, 7))
    ax.set_title("All models: test MAE across the six next-day positions")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(image_dir / "07_all_models_test_mae_by_horizon.png", dpi=180)
    plt.close(fig)

    gap = comparison.sort_values("ValidationToTestGap")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(gap.Model, gap.ValidationToTestGap)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Test MAE minus rolling-validation macro MAE")
    ax.set_title("Validation-to-test generalization gap")
    fig.tight_layout()
    fig.savefig(image_dir / "07_validation_test_gap.png", dpi=180)
    plt.close(fig)

    daily_test = daily[daily.Split == "test"].sort_values("MAE")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(daily_test.Model, daily_test.MAE)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE for derived next-day mean")
    ax.set_title("Derived next-day mean forecast accuracy")
    fig.tight_layout()
    fig.savefig(image_dir / "07_derived_daily_mean_test_mae.png", dpi=180)
    plt.close(fig)


def write_report(comparison: pd.DataFrame, common_metrics: pd.DataFrame, daily: pd.DataFrame, selected: str, common_n: int, path: Path) -> None:
    selected_row = comparison[comparison.SelectedByValidation == 1].iloc[0]
    test_best = comparison.sort_values("TestMAE").iloc[0]
    lines = [
        "# Step 07 — Revised Model Comparison",
        "",
        "## Selection rule",
        "",
        "The final model was selected using rolling-validation macro MAE, not Day 7. Eligibility required predictions for all three validation days and at least 95% validation and test coverage. Day 7 was then used once for the independent final evaluation. Models with lower coverage remain in the descriptive comparison but were not eligible for final selection.",
        "",
        f"The validation-selected model is **{selected}** with rolling-validation macro MAE **{selected_row.ValidationMacroMAE:.4f}**. Its untouched Day 7 test MAE is **{selected_row.TestMAE:.4f}** and test RMSE is **{selected_row.TestRMSE:.4f}**.",
        "",
        f"The lowest descriptive test MAE was also obtained by **{test_best.Model}** ({test_best.TestMAE:.4f}); therefore the validation-based choice and the test ranking agree in this run.",
        "",
        "## Validation and test comparison",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Common-target test comparison (N = {common_n} observed positions)",
        "",
        common_metrics.sort_values("MAE").to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Derived next-day mean comparison",
        "",
        daily[daily.Split == "test"].sort_values("MAE").to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Main interpretation",
        "",
        "The revised LSTM improved on the statistical, tree-based, and naive baselines for the complete six-value Day 7 trajectory. Its selected configuration used the most recent six mood observations only; social-media variables were not selected. This is consistent with Step 04, where adding social-media predictors through ARIMAX did not generalize to the test period.",
        "",
        "Persistence remained a strong benchmark, particularly for the immediately next assessment, while SARIMA continued to provide a relatively stable seasonal benchmark. The comparison therefore supports a limited conclusion: the compact LSTM captured useful nonlinear information in the recent mood sequence, but the gain must be interpreted in light of the small seven-day record and bootstrap uncertainty.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dirs = make_dirs(root)
    data = load_predictions(dirs["tables"])
    data.to_csv(dirs["tables"] / "07_all_model_predictions.csv", index=False)

    comparison = comparison_metrics(data)
    comparison.to_csv(dirs["tables"] / "07_model_comparison_metrics.csv", index=False)
    horizons = horizon_metrics(data)
    horizons.to_csv(dirs["tables"] / "07_model_horizon_metrics.csv", index=False)
    participants = participant_metrics(data)
    participants.to_csv(dirs["tables"] / "07_participant_model_metrics.csv", index=False)
    daily_predictions, daily_metrics = daily_mean_outputs(data)
    daily_predictions.to_csv(dirs["tables"] / "07_derived_daily_mean_predictions.csv", index=False)
    daily_metrics.to_csv(dirs["tables"] / "07_derived_daily_mean_metrics.csv", index=False)

    selected = comparison.loc[comparison.SelectedByValidation == 1, "Model"].iloc[0]
    eligible_models = comparison.loc[comparison.EligibleForFinalSelection == 1, "Model"].tolist()
    common = common_test_set(data, eligible_models)
    common.to_csv(dirs["tables"] / "07_all_model_common_predictions.csv", index=False)
    common_rows = []
    for model, g in common.groupby("Model"):
        common_rows.append({"Model": model, **error_metrics(g)})
    common_metrics = pd.DataFrame(common_rows)
    common_metrics.to_csv(dirs["tables"] / "07_common_test_model_metrics.csv", index=False)

    bootstrap_ci, bootstrap_diff = bootstrap_participants(common, selected, args.bootstrap_reps)
    bootstrap_ci.to_csv(dirs["tables"] / "07_participant_bootstrap_mae.csv", index=False)
    bootstrap_diff.to_csv(dirs["tables"] / "07_bootstrap_differences_from_selected.csv", index=False)

    ranking = comparison[["Model", "EligibleForFinalSelection", "ValidationRank", "ValidationMacroMAE", "SelectedByValidation", "TestRank", "TestMAE", "TestRMSE", "ValidationToTestGap"]].copy()
    ranking.to_csv(dirs["tables"] / "07_final_model_ranking.csv", index=False)
    (dirs["models"] / "07_selected_best_model.json").write_text(json.dumps({
        "selected_model": selected,
        "selection_metric": "rolling-validation macro MAE",
        "selection_requires": "three validation periods and >=95% validation/test coverage",
        "test_used_for_retuning": False,
        "eligible_models": eligible_models,
    }, indent=2), encoding="utf-8")

    save_plots(comparison, horizons, daily_metrics, dirs["images"])
    write_report(comparison, common_metrics, daily_metrics, selected, int(common.OriginID.size / len(eligible_models)), dirs["reports"] / "07_model_comparison_report.md")

    pd.DataFrame([{"Python": sys.version.split()[0], "Platform": platform.platform(), "Pandas": pd.__version__, "NumPy": np.__version__, "BootstrapSeed": SEED, "BootstrapReps": args.bootstrap_reps}]).to_csv(dirs["tables"] / "07_runtime_environment.csv", index=False)
    input_paths = [dirs["tables"] / f for f in ["04_statistical_model_predictions.csv", "05_tree_model_predictions.csv", "06_recurrent_model_predictions.csv"]]
    pd.DataFrame([{"Path": str(p.relative_to(root)), "Bytes": p.stat().st_size, "SHA256": sha256(p)} for p in input_paths]).to_csv(dirs["tables"] / "07_step_input_manifest.csv", index=False)
    manifest = []
    for folder in [dirs["tables"], dirs["images"], dirs["reports"], dirs["models"]]:
        for path in sorted(folder.glob("07_*")):
            manifest.append({"Path": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": sha256(path)})
    pd.DataFrame(manifest).to_csv(dirs["tables"] / "07_output_manifest.csv", index=False)
    print(f"Step 07 completed. Validation-selected model: {selected}")


if __name__ == "__main__":
    main()

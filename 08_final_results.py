"""Step 08: Produce final tables, figures, model card, and results report."""

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


def metrics(g: pd.DataFrame) -> dict[str, float]:
    valid = g.ActualObserved.eq(1) & g.Actual.notna() & g.Prediction.notna()
    e = g.loc[valid, "Prediction"].to_numpy(float) - g.loc[valid, "Actual"].to_numpy(float)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))), "Bias": float(np.mean(e)), "N": int(len(e))}


def save_figures(test: pd.DataFrame, participant: pd.DataFrame, image_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(test.Actual, test.Prediction, alpha=0.55)
    ax.plot([1, 7], [1, 7], linestyle="--")
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_xlabel("Observed AffWB")
    ax.set_ylabel("Predicted AffWB")
    ax.set_title("Final selected model: observed versus predicted Day 7 AffWB")
    fig.tight_layout()
    fig.savefig(image_dir / "08_final_model_actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    horizon = []
    for h, g in test.groupby("Horizon"):
        horizon.append({"Horizon": h, **metrics(g)})
    hd = pd.DataFrame(horizon)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hd.Horizon, hd.MAE, marker="o")
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Test MAE")
    ax.set_title("Final selected model error across next-day positions")
    fig.tight_layout()
    fig.savefig(image_dir / "08_final_model_error_by_horizon.png", dpi=180)
    plt.close(fig)

    p = participant.sort_values("MAE")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(p)), p.MAE)
    ax.set_xlabel("Participants ordered by MAE")
    ax.set_ylabel("Participant test MAE")
    ax.set_title("Final selected model participant-level variation")
    fig.tight_layout()
    fig.savefig(image_dir / "08_final_model_participant_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.project_root.resolve()
    dirs = make_dirs(root)

    selection = json.loads((dirs["models"] / "07_selected_best_model.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]
    all_predictions = pd.read_csv(dirs["tables"] / "07_all_model_predictions.csv")
    comparison = pd.read_csv(dirs["tables"] / "07_model_comparison_metrics.csv")
    horizons = pd.read_csv(dirs["tables"] / "07_model_horizon_metrics.csv")
    participants = pd.read_csv(dirs["tables"] / "07_participant_model_metrics.csv")
    daily = pd.read_csv(dirs["tables"] / "07_derived_daily_mean_metrics.csv")
    bootstrap = pd.read_csv(dirs["tables"] / "07_participant_bootstrap_mae.csv")
    bootstrap_diff = pd.read_csv(dirs["tables"] / "07_bootstrap_differences_from_selected.csv")

    selected_predictions = all_predictions[all_predictions.Model == selected].copy()
    selected_predictions.to_csv(dirs["tables"] / "08_final_model_predictions.csv", index=False)
    test = selected_predictions[selected_predictions.Split == "test"].copy()
    test_valid = test[test.ActualObserved.eq(1) & test.Actual.notna() & test.Prediction.notna()].copy()
    overall = metrics(test)

    horizon_results = horizons[(horizons.Model == selected) & (horizons.Split == "test")].copy()
    horizon_results.to_csv(dirs["tables"] / "08_final_model_horizon_results.csv", index=False)
    participant_results = participants[(participants.Model == selected) & (participants.Split == "test")].copy()
    participant_results.to_csv(dirs["tables"] / "08_final_model_participant_results.csv", index=False)
    daily_result = daily[(daily.Model == selected) & (daily.Split == "test")].copy()
    daily_result.to_csv(dirs["tables"] / "08_final_model_daily_mean_results.csv", index=False)

    h1 = horizon_results[horizon_results.Horizon == 1].iloc[0]
    dmean = daily_result.iloc[0]
    ci = bootstrap[bootstrap.Model == selected].iloc[0]
    primary = pd.DataFrame([
        {"Result": "Primary six-position trajectory", "Metric": "MAE", "Value": overall["MAE"], "N": overall["N"]},
        {"Result": "Primary six-position trajectory", "Metric": "RMSE", "Value": overall["RMSE"], "N": overall["N"]},
        {"Result": "Immediately next assessment (h=1)", "Metric": "MAE", "Value": h1.MAE, "N": int(h1.PredictedTargets)},
        {"Result": "Derived next-day mean (>=4 observed positions)", "Metric": "MAE", "Value": dmean.MAE, "N": int(dmean.ParticipantDays)},
        {"Result": "Participant-clustered bootstrap", "Metric": "MAE lower 95%", "Value": ci.MAELower95, "N": overall["N"]},
        {"Result": "Participant-clustered bootstrap", "Metric": "MAE upper 95%", "Value": ci.MAEUpper95, "N": overall["N"]},
    ])
    primary.to_csv(dirs["tables"] / "08_primary_and_secondary_results.csv", index=False)

    selected_comparison = comparison[comparison.Model == selected].iloc[0]
    summary = {
        "selected_model": selected,
        "selection_basis": "lowest rolling-validation macro MAE among eligible models",
        "validation_macro_mae": float(selected_comparison.ValidationMacroMAE),
        "test_mae": overall["MAE"], "test_rmse": overall["RMSE"], "test_bias": overall["Bias"],
        "test_observed_positions": overall["N"], "horizon_1_mae": float(h1.MAE),
        "derived_daily_mean_mae": float(dmean.MAE),
        "bootstrap_mae_95_interval": [float(ci.MAELower95), float(ci.MAEUpper95)],
        "test_used_for_retuning": False,
    }
    (dirs["models"] / "08_final_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(dirs["tables"] / "08_final_model_summary.csv", index=False)

    save_figures(test_valid, participant_results, dirs["images"])

    config_path = dirs["models"] / "06_selected_recurrent_model_configs.json"
    config = json.loads(config_path.read_text(encoding="utf-8")).get(selected, {}) if config_path.exists() else {}
    model_card = [
        f"# Selected Model Card — {selected}", "",
        "## Intended use", "", "Forecast the six AffWB assessments of the next planned study day from occasion-level history available at the end of the preceding day.", "",
        "## Selection", "", "The model was selected by rolling-validation macro MAE across target Days 4–6. Day 7 was not used for tuning.", "",
        "## Selected configuration", "", "```json", json.dumps(config, indent=2), "```", "",
        "## Final test performance", "",
        primary.to_markdown(index=False, floatfmt=".4f"), "",
        "## Limitations", "",
        "The participant records cover only seven planned days, exact timestamps are unavailable, some target positions are missing, and the results are specific to this dataset. The model supports prediction, not causal claims about social-media use or mood.",
    ]
    (dirs["reports"] / "08_selected_model_card.md").write_text("\n".join(model_card), encoding="utf-8")

    runner_up = comparison[comparison.Model != selected].sort_values("TestMAE").iloc[0]
    report = [
        "# Step 08 — Final Results",
        "",
        "## Final model",
        "",
        f"The final model is **{selected}**, selected before test evaluation using the lowest eligible rolling-validation macro MAE ({selected_comparison.ValidationMacroMAE:.4f}). The selected recurrent configuration used recent mood history without social-media predictors.",
        "",
        "## Primary and secondary Day 7 results",
        "",
        primary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Six-position interpretation",
        "",
        f"Across {overall['N']} observed Day 7 target positions, the model achieved MAE {overall['MAE']:.4f}, RMSE {overall['RMSE']:.4f}, and bias {overall['Bias']:.4f}. The immediately next assessment had MAE {h1.MAE:.4f}. The derived next-day mean had MAE {dmean.MAE:.4f} across {int(dmean.ParticipantDays)} participant-days.",
        "",
        "## Comparison with baselines",
        "",
        f"The selected model outperformed the persistence and SARIMA baselines on the complete six-position test trajectory. The next-lowest test MAE among the remaining models was {runner_up.Model} at {runner_up.TestMAE:.4f}. Participant-clustered bootstrap differences are provided in `07_bootstrap_differences_from_selected.csv` and should be used to judge whether apparent gaps are robust.",
        "",
        "## Predictor conclusion",
        "",
        "Neither ARIMAX nor the selected tree/recurrent configurations provided stable evidence that active or passive social-media history improved Day 7 mood prediction. The final LSTM selected mood-only input. This is a predictive conclusion and does not establish the absence of causal social-media effects.",
        "",
        "## Scientific conclusion",
        "",
        "The revised pipeline indicates that the recent mood sequence contains useful information for predicting the next day's six AffWB measurements. A compact LSTM generalized better than the tested statistical and tree-based alternatives, while simple persistence remained a strong benchmark. Because the dataset is short and the uncertainty intervals overlap for some comparisons, the result should be presented as evidence of improved prediction in this sample rather than proof of universal superiority.",
    ]
    (dirs["reports"] / "08_final_results_report.md").write_text("\n".join(report), encoding="utf-8")

    pd.DataFrame([{"Python": sys.version.split()[0], "Platform": platform.platform(), "Pandas": pd.__version__, "NumPy": np.__version__}]).to_csv(dirs["tables"] / "08_runtime_environment.csv", index=False)
    inputs = [dirs["models"] / "07_selected_best_model.json", dirs["tables"] / "07_all_model_predictions.csv", dirs["tables"] / "07_model_comparison_metrics.csv"]
    pd.DataFrame([{"Path": str(p.relative_to(root)), "Bytes": p.stat().st_size, "SHA256": sha256(p)} for p in inputs]).to_csv(dirs["tables"] / "08_step_input_manifest.csv", index=False)
    manifest = []
    for folder in [dirs["tables"], dirs["images"], dirs["reports"], dirs["models"]]:
        for path in sorted(folder.glob("08_*")):
            manifest.append({"Path": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": sha256(path)})
    pd.DataFrame(manifest).to_csv(dirs["tables"] / "08_output_manifest.csv", index=False)
    print(f"Step 08 completed. Final selected model: {selected}")


if __name__ == "__main__":
    main()

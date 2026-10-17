#!/usr/bin/env python3
"""Hierarchical, cluster-bootstrap, and candidate-resampling robustness analyses."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import (
    CONFIG,
    OUTPUTS_DIR,
    RESULTS_DIR,
    aggregate_metrics,
    bootstrap_ci,
    candidate_level_logits,
    ensure_dirs,
    layer_bands,
    logsumexp_np,
    mean_band,
    model_label,
    save_table,
)


def stem_late_minus_middle(metrics: pd.DataFrame, probe_name: str, condition: str | None = None) -> pd.DataFrame:
    selected = metrics[metrics["probe_name"].eq(probe_name)].copy()
    if condition is not None:
        selected = selected[selected["condition"].eq(condition)]
    rows = []
    for keys, group in selected.groupby(["model_id", "prompt_id"], sort=False):
        model_id, prompt_id = keys
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        table = group.pivot_table(index="prompt_id", columns="layer_index", values="margin", aggfunc="mean")
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label(model_id),
                "probe_name": probe_name,
                "unit_id": prompt_id,
                "late_minus_middle": float((mean_band(table, bands["late"]) - mean_band(table, bands["middle"])).iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def self_advantage_unit_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["probe_name"].eq("self_object")].copy()
    rows = []
    for keys, group in selected.groupby(["model_id", "pair_id"], sort=False):
        model_id, pair_id = keys
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        table = group.pivot_table(index="condition", columns="layer_index", values="margin", aggfunc="mean")
        if {"self", "object"}.issubset(table.index):
            self_delta = mean_band(table.loc[["self"]], bands["late"]).iloc[0] - mean_band(table.loc[["self"]], bands["middle"]).iloc[0]
            object_delta = mean_band(table.loc[["object"]], bands["late"]).iloc[0] - mean_band(table.loc[["object"]], bands["middle"]).iloc[0]
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label(model_id),
                    "probe_name": "self_object_advantage",
                    "unit_id": pair_id,
                    "late_minus_middle": float(self_delta - object_delta),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap(df: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    models = np.array(sorted(df["model_label"].unique()))
    values = np.empty(n, dtype=float)
    grouped = {model: group for model, group in df.groupby("model_label", sort=False)}
    for i in range(n):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        sampled_values = []
        for model in sampled_models:
            group = grouped[model]
            idx = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
            sampled_values.extend(group.loc[idx, "late_minus_middle"].to_numpy(dtype=float).tolist())
        values[i] = float(np.mean(sampled_values))
    return values


def summary_estimates(df: pd.DataFrame, probe_name: str) -> list[dict]:
    rows = []
    rows.append(
        {
            "probe_name": probe_name,
            "estimator": "naive_grand_mean",
            "estimate": float(df["late_minus_middle"].mean()),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_models": int(df["model_label"].nunique()),
            "n_units": int(df["unit_id"].nunique()),
            "warning": "",
        }
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mixed = smf.mixedlm("late_minus_middle ~ 1", data=df, groups=df["model_label"]).fit(reml=True, method="lbfgs")
        rows.append(
            {
                "probe_name": probe_name,
                "estimator": "random_model_intercept_mixedlm",
                "estimate": float(mixed.params["Intercept"]),
                "ci_low": float(mixed.conf_int().loc["Intercept", 0]),
                "ci_high": float(mixed.conf_int().loc["Intercept", 1]),
                "n_models": int(df["model_label"].nunique()),
                "n_units": int(df["unit_id"].nunique()),
                "warning": "; ".join(str(w.message) for w in caught),
            }
        )
    except Exception as exc:
        rows.append(
            {
                "probe_name": probe_name,
                "estimator": "random_model_intercept_mixedlm",
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_models": int(df["model_label"].nunique()),
                "n_units": int(df["unit_id"].nunique()),
                "warning": repr(exc),
            }
        )
    return rows


def leave_one_model(df: pd.DataFrame, probe_name: str) -> pd.DataFrame:
    rows = []
    for left_out in sorted(df["model_label"].unique()):
        kept = df[~df["model_label"].eq(left_out)]
        rows.append(
            {
                "probe_name": probe_name,
                "left_out_model": left_out,
                "grand_mean_without_model": float(kept["late_minus_middle"].mean()),
                "models_remaining": int(kept["model_label"].nunique()),
                "units_remaining": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def probe_unit_tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {
        "self_object_advantage": self_advantage_unit_table(metrics),
        "reflective_practical": stem_late_minus_middle(metrics, "reflective_practical"),
        "cova_verb_noun": stem_late_minus_middle(metrics, "cova_verb_noun"),
    }
    return tables


def tensor_from_words(word_df: pd.DataFrame, unit_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[int]]:
    keys = unit_cols + ["layer_index"]
    pivot = word_df.pivot_table(index=keys, columns="candidate_text", values="word_logit", aggfunc="mean")
    meta = word_df.drop_duplicates(["candidate_text"])[["candidate_text", "analysis_group"]].set_index("candidate_text")
    pivot = pivot[meta.index.tolist()]
    index = pivot.index.to_frame(index=False)
    layers = sorted(index["layer_index"].unique())
    unit_index = index[unit_cols].drop_duplicates().reset_index(drop=True)
    arr = pivot.to_numpy().reshape(len(unit_index), len(layers), len(meta))
    return arr, meta.reset_index(), unit_index, [int(layer) for layer in layers]


def band_delta_from_tensor(arr: np.ndarray, layers: list[int], target_idx: np.ndarray, contrast_idx: np.ndarray) -> np.ndarray:
    n_layers = max(layers)
    bands = layer_bands(n_layers)
    layer_to_pos = {layer: pos for pos, layer in enumerate(layers)}
    middle_pos = [layer_to_pos[layer] for layer in bands["middle"] if layer in layer_to_pos]
    late_pos = [layer_to_pos[layer] for layer in bands["late"] if layer in layer_to_pos]
    margins = logsumexp_np(arr[:, :, target_idx], axis=2) - logsumexp_np(arr[:, :, contrast_idx], axis=2)
    return np.nanmean(margins[:, late_pos], axis=1) - np.nanmean(margins[:, middle_pos], axis=1)


def self_object_stat_from_tensor(arr: np.ndarray, layers: list[int], unit_index: pd.DataFrame, target_idx: np.ndarray, contrast_idx: np.ndarray) -> float:
    delta = band_delta_from_tensor(arr, layers, target_idx, contrast_idx)
    tmp = unit_index.copy()
    tmp["delta"] = delta
    wide = tmp.pivot_table(index="pair_id", columns="condition", values="delta", aggfunc="mean")
    if not {"self", "object"}.issubset(wide.columns):
        return np.nan
    return float(np.nanmean(wide["self"] - wide["object"]))


def simple_stat_from_tensor(arr: np.ndarray, layers: list[int], target_idx: np.ndarray, contrast_idx: np.ndarray) -> float:
    return float(np.nanmean(band_delta_from_tensor(arr, layers, target_idx, contrast_idx)))


def candidate_bootstrap_and_leave_one(words: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    boot_rows = []
    leave_rows = []
    for (model_id, probe_name), group in words.groupby(["model_id", "probe_name"], sort=False):
        if probe_name not in {"reflective_practical", "self_object", "cova_verb_noun"}:
            continue
        probe_label = "self_object_advantage" if probe_name == "self_object" else probe_name
        strata = [("all", group)] if probe_name != "cova_verb_noun" else list(group.groupby("syntactic_frame", sort=False))
        tensors = []
        for _stratum_name, stratum in strata:
            unit_cols = ["pair_id", "condition"] if probe_name == "self_object" else ["prompt_id"]
            arr, meta, unit_index, layers = tensor_from_words(stratum, unit_cols)
            target_idx = np.where(meta["analysis_group"].eq("target").to_numpy())[0]
            contrast_idx = np.where(meta["analysis_group"].eq("contrast").to_numpy())[0]
            if len(target_idx) < 2 or len(contrast_idx) < 2:
                continue
            tensors.append((arr, meta, unit_index, layers, target_idx, contrast_idx))
        if not tensors:
            continue
        for i in range(n_boot):
            values = []
            weights = []
            for arr, _meta, unit_index, layers, target_idx, contrast_idx in tensors:
                sampled_target = rng.choice(target_idx, size=len(target_idx), replace=True)
                sampled_contrast = rng.choice(contrast_idx, size=len(contrast_idx), replace=True)
                if probe_name == "self_object":
                    values.append(self_object_stat_from_tensor(arr, layers, unit_index, sampled_target, sampled_contrast))
                    weights.append(1)
                else:
                    deltas = band_delta_from_tensor(arr, layers, sampled_target, sampled_contrast)
                    values.append(float(np.nanmean(deltas)))
                    weights.append(int(np.sum(~np.isnan(deltas))))
            boot_rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label(model_id),
                    "probe_name": probe_label,
                    "bootstrap_index": i,
                    "late_minus_middle": float(np.average(values, weights=weights)),
                }
            )
        seen_words = group.drop_duplicates(["candidate_text", "analysis_group"])[["candidate_text", "analysis_group"]]
        for _, word_row in seen_words.iterrows():
            word = word_row["candidate_text"]
            values = []
            weights = []
            for arr, meta, unit_index, layers, target_idx, contrast_idx in tensors:
                word_cols = np.where(meta["candidate_text"].eq(word).to_numpy())[0]
                remove_col = int(word_cols[0]) if len(word_cols) else None
                t_idx = np.array([idx for idx in target_idx if idx != remove_col], dtype=int)
                c_idx = np.array([idx for idx in contrast_idx if idx != remove_col], dtype=int)
                if len(t_idx) == 0 or len(c_idx) == 0:
                    continue
                if probe_name == "self_object":
                    values.append(self_object_stat_from_tensor(arr, layers, unit_index, t_idx, c_idx))
                    weights.append(1)
                else:
                    deltas = band_delta_from_tensor(arr, layers, t_idx, c_idx)
                    values.append(float(np.nanmean(deltas)))
                    weights.append(int(np.sum(~np.isnan(deltas))))
            if values:
                leave_rows.append(
                    {
                        "model_id": model_id,
                        "model_label": model_label(model_id),
                        "probe_name": probe_label,
                        "left_out_candidate": word,
                        "left_out_group": word_row["analysis_group"],
                        "late_minus_middle": float(np.average(values, weights=weights)),
                    }
                )
    return pd.DataFrame(boot_rows), pd.DataFrame(leave_rows)


def continuous_depth_contrast(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = metrics[metrics["probe_name"].eq("self_object")].copy()
    rows = []
    for keys, group in selected.groupby(["model_id", "pair_id", "layer_index", "relative_depth"], sort=False):
        model_id, pair_id, layer_index, relative_depth = keys
        table = group.pivot_table(index="condition", values="margin", aggfunc="mean")
        if {"self", "object"}.issubset(table.index):
            rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label(model_id),
                    "pair_id": pair_id,
                    "layer_index": int(layer_index),
                    "relative_depth": float(relative_depth),
                    "self_minus_object_margin": float(table.loc["self", "margin"] - table.loc["object", "margin"]),
                }
            )
    contrast = pd.DataFrame(rows)
    contrast["relative_depth_sq"] = contrast["relative_depth"] ** 2
    model = smf.ols("self_minus_object_margin ~ relative_depth + relative_depth_sq + C(model_label)", data=contrast).fit()
    grid = pd.DataFrame({"relative_depth": np.linspace(0, 1, 101)})
    grid["relative_depth_sq"] = grid["relative_depth"] ** 2
    grid["model_label"] = "gpt2"
    grid["predicted_self_minus_object_margin"] = model.predict(grid)
    params = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
            "p_value": model.pvalues.values,
        }
    )
    return contrast, pd.concat([grid.assign(row_type="prediction"), params.assign(row_type="coefficient")], ignore_index=True, sort=False)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]))
    n_boot = int(CONFIG["candidate_bootstrap_resamples"])
    metrics = aggregate_metrics()
    words = candidate_level_logits()

    tables = probe_unit_tables(metrics)
    all_units = pd.concat(tables.values(), ignore_index=True, sort=False)
    save_table(all_units, RESULTS_DIR / "hierarchical_unit_effects")

    estimate_rows = []
    loo_frames = []
    bootstrap_rows = []
    for probe_name, table in tables.items():
        estimate_rows.extend(summary_estimates(table, probe_name))
        boot = cluster_bootstrap(table, rng, n_boot)
        ci_low, ci_high = bootstrap_ci(boot)
        estimate_rows.append(
            {
                "probe_name": probe_name,
                "estimator": "cluster_bootstrap_models_and_units",
                "estimate": float(table["late_minus_middle"].mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_models": int(table["model_label"].nunique()),
                "n_units": int(table["unit_id"].nunique()),
                "warning": "",
            }
        )
        bootstrap_rows.append(
            pd.DataFrame(
                {
                    "probe_name": probe_name,
                    "bootstrap_index": np.arange(n_boot),
                    "grand_mean": boot,
                }
            )
        )
        loo_frames.append(leave_one_model(table, probe_name))

    estimates = pd.DataFrame(estimate_rows)
    save_table(estimates, RESULTS_DIR / "hierarchical_estimates")
    save_table(pd.concat(loo_frames, ignore_index=True), RESULTS_DIR / "leave_one_model_out")
    save_table(pd.concat(bootstrap_rows, ignore_index=True), RESULTS_DIR / "hierarchical_cluster_bootstrap")

    candidate_boot, leave_candidate = candidate_bootstrap_and_leave_one(words, rng, n_boot)
    save_table(candidate_boot, RESULTS_DIR / "candidate_bootstrap")
    save_table(leave_candidate, RESULTS_DIR / "leave_one_candidate_out")

    cova_dimension = pd.read_csv(OUTPUTS_DIR / "posthoc_cova_dimension_robustness.csv")
    cova_dimension["positive_after_partial_pooling_proxy"] = cova_dimension["late_minus_middle"] > 0
    save_table(cova_dimension, RESULTS_DIR / "cova_dimension_partial_pooling_proxy")

    contrast, continuous = continuous_depth_contrast(metrics)
    save_table(contrast, RESULTS_DIR / "continuous_self_object_layer_contrast_observations")
    save_table(continuous, RESULTS_DIR / "continuous_depth_contrast")

    report = [
        "# Hierarchical Inference Report",
        "",
        "Formal Bayesian models were not used. `statsmodels` mixed-effects models were attempted where stable, and all main estimates are paired with model/stimulus cluster bootstraps. Candidate words were also resampled within target/control class; this is a robustness check over sampled words, not a substitute for future human norming.",
        "",
        "## Main Estimates",
        "",
        estimates.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Candidate Bootstrap",
        "",
        "Candidate bootstrap rows were saved to `results/robustness/candidate_bootstrap.*`; leave-one-candidate diagnostics were saved to `results/robustness/leave_one_candidate_out.*`.",
        "",
        "## Continuous Layer Model",
        "",
        "A prespecified quadratic model of the self-minus-object contrast over normalized depth was fit as a low-complexity continuous alternative to hard layer bands. Coefficients and predictions are saved in `results/robustness/continuous_depth_contrast.*`.",
        "",
    ]
    (RESULTS_DIR.parent.parent / "reports" / "hierarchical_inference_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Wrote hierarchical, bootstrap, and candidate-resampling outputs.")


if __name__ == "__main__":
    main()

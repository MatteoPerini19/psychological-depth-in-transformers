#!/usr/bin/env python3
"""Layer-window and peak-location sensitivity analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    CONFIG,
    RESULTS_DIR,
    aggregate_metrics,
    bootstrap_ci,
    ensure_dirs,
    layer_bands,
    layer_region,
    mean_band,
    model_label,
    save_table,
)


def enumerate_windows(n_layers: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    candidate_layers = list(range(1, n_layers))
    widths = [2] if n_layers <= 6 else [2, 3, 4]
    middle = []
    late = []
    for width in widths:
        for start in candidate_layers:
            window = tuple(range(start, start + width))
            if window[-1] >= n_layers:
                continue
            center = np.mean(window) / n_layers
            if 0.25 <= center <= 0.60:
                middle.append(window)
            if 0.60 <= center <= 0.92:
                late.append(window)
    middle = sorted(set(middle))
    late = sorted(set(late))
    return middle, late


def unit_layer_table(metrics: pd.DataFrame, model_id: str, probe_name: str) -> pd.DataFrame:
    if probe_name == "self_object_advantage":
        selected = metrics[(metrics["model_id"].eq(model_id)) & (metrics["probe_name"].eq("self_object"))]
        rows = []
        for keys, group in selected.groupby(["pair_id", "layer_index", "relative_depth"], sort=False):
            pair_id, layer_index, relative_depth = keys
            table = group.pivot_table(index="condition", values="margin", aggfunc="mean")
            if {"self", "object"}.issubset(table.index):
                rows.append(
                    {
                        "unit_id": pair_id,
                        "layer_index": int(layer_index),
                        "relative_depth": float(relative_depth),
                        "margin": float(table.loc["self", "margin"] - table.loc["object", "margin"]),
                    }
                )
        return pd.DataFrame(rows)
    selected = metrics[(metrics["model_id"].eq(model_id)) & (metrics["probe_name"].eq(probe_name))]
    return selected.rename(columns={"prompt_id": "unit_id"})[["unit_id", "layer_index", "relative_depth", "margin"]].copy()


def effect_for_windows(table: pd.DataFrame, middle_window: tuple[int, ...], late_window: tuple[int, ...]) -> float:
    pivot = table.pivot_table(index="unit_id", columns="layer_index", values="margin", aggfunc="mean")
    middle = pivot[[layer for layer in middle_window if layer in pivot.columns]].mean(axis=1)
    late = pivot[[layer for layer in late_window if layer in pivot.columns]].mean(axis=1)
    return float((late - middle).mean())


def primary_effect(table: pd.DataFrame, n_layers: int) -> float:
    bands = layer_bands(n_layers)
    return effect_for_windows(table, tuple(bands["middle"]), tuple(bands["late"]))


def peak_bootstrap(table: pd.DataFrame, n_layers: int, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    pivot = table.pivot_table(index="unit_id", columns="layer_index", values="margin", aggfunc="mean")
    layer_cols = [col for col in sorted(pivot.columns) if col != 0]
    units = pivot.index.to_numpy()
    rows = []
    for i in range(n_boot):
        sampled = rng.choice(units, size=len(units), replace=True)
        curve = pivot.loc[sampled, layer_cols].mean(axis=0)
        peak_layer = int(curve.idxmax())
        rows.append(
            {
                "bootstrap_index": i,
                "peak_layer": peak_layer,
                "peak_relative_depth": peak_layer / n_layers,
                "peak_region": layer_region(peak_layer, n_layers),
                "late_exceeds_final": float(curve[[layer for layer in layer_bands(n_layers)["late"] if layer in curve.index]].mean()) > float(curve.loc[n_layers]) if n_layers in curve.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def leave_one_layer(primary_table: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    bands = layer_bands(n_layers)
    rows = []
    for band_name in ["middle", "late"]:
        for layer in bands[band_name]:
            middle = [x for x in bands["middle"] if not (band_name == "middle" and x == layer)]
            late = [x for x in bands["late"] if not (band_name == "late" and x == layer)]
            if not middle or not late:
                continue
            rows.append(
                {
                    "left_out_band": band_name,
                    "left_out_layer": layer,
                    "effect": effect_for_windows(primary_table, tuple(middle), tuple(late)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]) + 11)
    n_boot = int(CONFIG["peak_bootstrap_resamples"])
    metrics = aggregate_metrics()
    probes = ["reflective_practical", "self_object_advantage", "cova_verb_noun"]
    grid_rows = []
    summary_rows = []
    peak_frames = []
    loo_frames = []

    for model_id in CONFIG["models"]:
        n_layers = int(metrics[metrics["model_id"].eq(model_id)]["layer_index"].max())
        middle_windows, late_windows = enumerate_windows(n_layers)
        for probe_name in probes:
            table = unit_layer_table(metrics, model_id, probe_name)
            primary = primary_effect(table, n_layers)
            effects = []
            for middle_window in middle_windows:
                for late_window in late_windows:
                    if max(middle_window) >= min(late_window):
                        continue
                    effect = effect_for_windows(table, middle_window, late_window)
                    effects.append(effect)
                    grid_rows.append(
                        {
                            "model_id": model_id,
                            "model_label": model_label(model_id),
                            "probe_name": probe_name,
                            "n_transformer_layers": n_layers,
                            "middle_window": " ".join(map(str, middle_window)),
                            "late_window": " ".join(map(str, late_window)),
                            "middle_center_relative": float(np.mean(middle_window) / n_layers),
                            "late_center_relative": float(np.mean(late_window) / n_layers),
                            "effect": effect,
                            "same_sign_as_primary": np.sign(effect) == np.sign(primary),
                            "is_primary_specification": tuple(layer_bands(n_layers)["middle"]) == middle_window
                            and tuple(layer_bands(n_layers)["late"]) == late_window,
                        }
                    )
            effects_arr = np.asarray(effects, dtype=float)
            summary_rows.append(
                {
                    "model_id": model_id,
                    "model_label": model_label(model_id),
                    "probe_name": probe_name,
                    "n_transformer_layers": n_layers,
                    "n_window_specifications": int(len(effects_arr)),
                    "primary_effect": primary,
                    "proportion_positive": float(np.mean(effects_arr > 0)),
                    "proportion_same_sign_as_primary": float(np.mean(np.sign(effects_arr) == np.sign(primary))),
                    "median_effect": float(np.median(effects_arr)),
                    "min_effect": float(np.min(effects_arr)),
                    "max_effect": float(np.max(effects_arr)),
                    "q10_effect": float(np.quantile(effects_arr, 0.10)),
                    "q90_effect": float(np.quantile(effects_arr, 0.90)),
                }
            )
            peaks = peak_bootstrap(table, n_layers, rng, n_boot)
            peaks["model_id"] = model_id
            peaks["model_label"] = model_label(model_id)
            peaks["probe_name"] = probe_name
            peak_frames.append(peaks)
            loo = leave_one_layer(table, n_layers)
            loo["model_id"] = model_id
            loo["model_label"] = model_label(model_id)
            loo["probe_name"] = probe_name
            loo_frames.append(loo)

    grid = pd.DataFrame(grid_rows)
    summary = pd.DataFrame(summary_rows)
    peak = pd.concat(peak_frames, ignore_index=True)
    peak_summary = (
        peak.groupby(["model_id", "model_label", "probe_name"], sort=False)
        .agg(
            peak_relative_mean=("peak_relative_depth", "mean"),
            peak_relative_ci_low=("peak_relative_depth", lambda x: bootstrap_ci(np.asarray(x))[0]),
            peak_relative_ci_high=("peak_relative_depth", lambda x: bootstrap_ci(np.asarray(x))[1]),
            prob_peak_late_intermediate=("peak_region", lambda x: float(np.mean(x == "late_intermediate"))),
            prob_peak_final=("peak_region", lambda x: float(np.mean(x == "final"))),
            prob_late_exceeds_final=("late_exceeds_final", "mean"),
        )
        .reset_index()
    )
    save_table(grid, RESULTS_DIR / "layer_window_grid")
    save_table(summary, RESULTS_DIR / "layer_window_summary")
    save_table(peak, RESULTS_DIR / "peak_bootstrap")
    save_table(peak_summary, RESULTS_DIR / "peak_bootstrap_summary")
    save_table(pd.concat(loo_frames, ignore_index=True), RESULTS_DIR / "leave_one_layer_out")

    report = [
        "# Layer-Window Sensitivity Report",
        "",
        "Window specifications were enumerated before inspecting effects. Embedding states were excluded from all window grids, and final block outputs were treated separately from late-intermediate windows.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Peak Bootstrap Summary",
        "",
        peak_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    (RESULTS_DIR.parent.parent / "reports" / "layer_window_sensitivity_report.md").write_text("\n".join(report), encoding="utf-8")
    print("Wrote layer-window sensitivity outputs.")


if __name__ == "__main__":
    main()

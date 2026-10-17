#!/usr/bin/env python3
"""Validate the canonical robustness dataset against manuscript outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUTPUTS_DIR, REPORTS_DIR, RESULTS_DIR, aggregate_metrics, ensure_dirs, layer_bands, mean_band


TOL = 1e-9


def recompute_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    main = metrics.rename(columns={"model_id": "model_name", "stem_text": "stem"})
    derived = []
    for (_model_name, _probe_name), group in main.groupby(["model_name", "probe_name"], sort=False):
        if group["condition"].nunique(dropna=False) > 1:
            copied = group.copy()
            copied["condition"] = "all_stems"
            derived.append(copied)
    if derived:
        main = pd.concat([main] + derived, ignore_index=True, sort=False)
    for keys, group in main.groupby(["model_name", "probe_name", "condition"], dropna=False, sort=False):
        model_name, probe_name, condition = keys
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        table = group.pivot_table(index="prompt_id", columns="layer_index", values="margin", aggfunc="mean")
        middle = mean_band(table, bands["middle"])
        late = mean_band(table, bands["late"])
        final = mean_band(table, bands["final"])
        rows.append(
            {
                "model_name": model_name,
                "probe_name": probe_name,
                "stem_condition": condition,
                "middle_mean_margin_recomputed": float(middle.mean()),
                "late_mean_margin_recomputed": float(late.mean()),
                "late_minus_middle_recomputed": float((late - middle).mean()),
                "final_mean_margin_recomputed": float(final.mean()),
                "n_stems_recomputed": int(len(table)),
            }
        )
    return pd.DataFrame(rows)


def validate_self_advantage(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["probe_name"].eq("self_object")].copy()
    rows = []
    for (model_name, pair_id), group in selected.groupby(["model_id", "pair_id"], sort=False):
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        pivot = group.pivot_table(
            index=["condition"],
            columns="layer_index",
            values="margin",
            aggfunc="mean",
        )
        if {"self", "object"}.issubset(pivot.index):
            self_middle = mean_band(pivot.loc[["self"]], bands["middle"]).iloc[0]
            self_late = mean_band(pivot.loc[["self"]], bands["late"]).iloc[0]
            self_final = mean_band(pivot.loc[["self"]], bands["final"]).iloc[0]
            obj_middle = mean_band(pivot.loc[["object"]], bands["middle"]).iloc[0]
            obj_late = mean_band(pivot.loc[["object"]], bands["late"]).iloc[0]
            obj_final = mean_band(pivot.loc[["object"]], bands["final"]).iloc[0]
            rows.append(
                {
                    "model_name": model_name,
                    "pair_id": pair_id,
                    "probe_name": "self_object_advantage",
                    "stem_condition": "self_minus_object_stems",
                    "middle_margin": self_middle - obj_middle,
                    "late_margin": self_late - obj_late,
                    "late_minus_middle": (self_late - obj_late) - (self_middle - obj_middle),
                    "final_margin": self_final - obj_final,
                }
            )
    pair_rows = pd.DataFrame(rows)
    return (
        pair_rows.groupby(["model_name", "probe_name", "stem_condition"], sort=False)
        .agg(
            late_minus_middle_recomputed=("late_minus_middle", "mean"),
            final_mean_margin_recomputed=("final_margin", "mean"),
            n_stems_recomputed=("pair_id", "nunique"),
        )
        .reset_index()
    )


def main() -> None:
    ensure_dirs()
    checks = []

    data_counts = {
        "simple_word_group_stems.csv": 42,
        "self_object_stems.csv": 72,
        "self_object_word_groups.csv": 20,
        "cova_verb_noun_stems.csv": 66,
        "cova_depth_completion_words_single_token.csv": 109,
    }
    for file_name, expected in data_counts.items():
        path = RESULTS_DIR.parent.parent / "data" / file_name
        actual = len(pd.read_csv(path))
        checks.append({"check": f"row_count::{file_name}", "expected": expected, "actual": actual, "passed": actual == expected})

    token_path = OUTPUTS_DIR / "cross_model_sentence_completion_token_check.csv"
    token = pd.read_csv(token_path)
    checks.append(
        {
            "check": "main_token_check_usable_count",
            "expected": 620,
            "actual": int(token["is_usable"].sum()),
            "passed": int(token["is_usable"].sum()) == 620,
        }
    )

    metrics = aggregate_metrics()
    max_layers = metrics.groupby("model_id")["layer_index"].max().astype(int).to_dict()
    expected_layers = {
        "gpt2": 12,
        "distilgpt2": 6,
        "EleutherAI/pythia-14m": 6,
        "EleutherAI/pythia-70m": 6,
        "EleutherAI/pythia-70m-deduped": 6,
        "EleutherAI/pythia-160m": 12,
    }
    for model_name, expected in expected_layers.items():
        actual = int(max_layers.get(model_name, -1))
        checks.append({"check": f"layer_count::{model_name}", "expected": expected, "actual": actual, "passed": actual == expected})

    original = pd.read_csv(OUTPUTS_DIR / "cross_model_sentence_completion_summary_by_model.csv")
    recomputed = recompute_summary(metrics)
    self_adv = validate_self_advantage(metrics)
    recomputed_all = pd.concat([recomputed, self_adv], ignore_index=True, sort=False)
    merged = original.merge(
        recomputed_all,
        on=["model_name", "probe_name", "stem_condition"],
        how="left",
        validate="one_to_one",
    )
    for _, row in merged.iterrows():
        for column, recomputed_column in [
            ("late_minus_middle", "late_minus_middle_recomputed"),
            ("final_mean_margin", "final_mean_margin_recomputed"),
        ]:
            if pd.isna(row[recomputed_column]):
                passed = False
                diff = np.nan
            else:
                diff = abs(float(row[column]) - float(row[recomputed_column]))
                passed = diff < TOL
            checks.append(
                {
                    "check": f"summary::{row['model_label']}::{row['probe_name']}::{row['stem_condition']}::{column}",
                    "expected": float(row[column]) if pd.notna(row[column]) else np.nan,
                    "actual": float(row[recomputed_column]) if pd.notna(row[recomputed_column]) else np.nan,
                    "difference": diff,
                    "passed": bool(passed),
                }
            )

    if (OUTPUTS_DIR / "causal_patch_late_component_summary.csv").exists():
        patch = pd.read_csv(OUTPUTS_DIR / "causal_patch_late_component_summary.csv")
        checks.append(
            {
                "check": "existing_patching_rows",
                "expected": 24,
                "actual": len(patch),
                "passed": len(patch) == 24,
            }
        )
        checks.append(
            {
                "check": "existing_resid_post_models_positive",
                "expected": 6,
                "actual": int(patch[patch["component"].eq("resid_post")]["positive_late_causal_effect"].sum()),
                "passed": int(patch[patch["component"].eq("resid_post")]["positive_late_causal_effect"].sum()) == 6,
            }
        )

    check_df = pd.DataFrame(checks)
    check_df.to_csv(RESULTS_DIR / "reproduction_validation_checks.csv", index=False)
    failed = check_df[~check_df["passed"].astype(bool)]
    report = [
        "# Reproduction Validation Report",
        "",
        f"Checks run: {len(check_df)}.",
        f"Checks passed: {int(check_df['passed'].sum())}.",
        f"Checks failed: {len(failed)}.",
        "",
        "The canonical robustness dataset preserves the existing layer convention: embedding is `layer_index = 0`, transformer block outputs are `1..N`, and the final block is summarized separately from late-intermediate layers.",
        "",
    ]
    if failed.empty:
        report.append("All reproduction checks passed within numerical tolerance.")
    else:
        report.append("Failed checks:")
        for _, row in failed.iterrows():
            report.append(f"- {row['check']}: expected `{row.get('expected')}`, actual `{row.get('actual')}`.")
    (REPORTS_DIR / "reproduction_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(check_df["passed"].sum(), "passed /", len(check_df), "checks")


if __name__ == "__main__":
    main()

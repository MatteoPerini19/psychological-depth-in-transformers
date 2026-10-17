#!/usr/bin/env python3
"""Large cached permutation tests for the sentence-completion probes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    CONFIG,
    RESULTS_DIR,
    aggregate_metrics,
    bh_adjust,
    candidate_level_logits,
    ensure_dirs,
    holm_adjust,
    layer_bands,
    logsumexp_np,
    mean_band,
    model_label as label_for_model,
    permutation_summary,
    save_table,
)


def band_stat_from_margin_frame(group: pd.DataFrame, unit_cols: list[str], margin_col: str = "margin") -> pd.Series:
    n_layers = int(group["layer_index"].max())
    bands = layer_bands(n_layers)
    table = group.pivot_table(index=unit_cols, columns="layer_index", values=margin_col, aggfunc="mean")
    middle = mean_band(table, bands["middle"])
    late = mean_band(table, bands["late"])
    return late - middle


def self_pair_deltas(metrics: pd.DataFrame, model_id: str) -> pd.Series:
    selected = metrics[(metrics["model_id"].eq(model_id)) & (metrics["probe_name"].eq("self_object"))]
    n_layers = int(selected["layer_index"].max())
    bands = layer_bands(n_layers)
    rows = []
    for pair_id, group in selected.groupby("pair_id", sort=False):
        table = group.pivot_table(index="condition", columns="layer_index", values="margin", aggfunc="mean")
        if {"self", "object"}.issubset(table.index):
            self_delta = mean_band(table.loc[["self"]], bands["late"]).iloc[0] - mean_band(table.loc[["self"]], bands["middle"]).iloc[0]
            obj_delta = mean_band(table.loc[["object"]], bands["late"]).iloc[0] - mean_band(table.loc[["object"]], bands["middle"]).iloc[0]
            rows.append({"pair_id": pair_id, "delta": self_delta - obj_delta})
    return pd.DataFrame(rows).set_index("pair_id")["delta"]


def sign_flip_null(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> np.ndarray:
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)))
    return (signs * values[None, :]).mean(axis=1)


def tensor_for_candidate_permutation(word_df: pd.DataFrame, unit_cols: list[str]) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = unit_cols + ["layer_index"]
    pivot = word_df.pivot_table(index=keys, columns="candidate_text", values="word_logit", aggfunc="mean")
    meta = word_df.drop_duplicates(["candidate_text"])[
        ["candidate_text", "analysis_group", "lexical_type", "syntactic_frame"]
    ].set_index("candidate_text")
    pivot = pivot[meta.index.tolist()]
    index = pivot.index.to_frame(index=False)
    layers = sorted(index["layer_index"].unique())
    unit_index = index[unit_cols].drop_duplicates().reset_index(drop=True)
    arr = pivot.to_numpy().reshape(len(unit_index), len(layers), len(meta))
    layer_to_pos = {int(layer): pos for pos, layer in enumerate(layers)}
    return (
        arr,
        meta.reset_index(),
        pd.DataFrame({"layer_index": layers, "layer_pos": [layer_to_pos[int(x)] for x in layers]}),
        unit_index,
    )


def late_minus_middle_from_arr(arr: np.ndarray, layer_df: pd.DataFrame, target_mask: np.ndarray, n_layers: int) -> np.ndarray:
    bands = layer_bands(n_layers)
    layer_to_pos = dict(zip(layer_df["layer_index"].astype(int), layer_df["layer_pos"].astype(int)))
    middle_pos = [layer_to_pos[layer] for layer in bands["middle"] if layer in layer_to_pos]
    late_pos = [layer_to_pos[layer] for layer in bands["late"] if layer in layer_to_pos]
    margins = logsumexp_np(arr[:, :, target_mask], axis=2) - logsumexp_np(arr[:, :, ~target_mask], axis=2)
    return margins[:, late_pos].mean(axis=1) - margins[:, middle_pos].mean(axis=1)


def candidate_permutation_null(
    word_df: pd.DataFrame,
    unit_cols: list[str],
    strata_cols: list[str],
    statistic: str,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, np.ndarray]:
    n_layers = int(word_df["layer_index"].max())
    if not strata_cols:
        strata_cols = ["_all"]
        word_df = word_df.copy()
        word_df["_all"] = "all"
    observed_parts = []
    null_parts = []
    for _, stratum in word_df.groupby(strata_cols, dropna=False, sort=False):
        arr, meta, layer_df, _unit_index = tensor_for_candidate_permutation(stratum, unit_cols)
        target_mask = meta["analysis_group"].eq("target").to_numpy()
        observed_parts.append(late_minus_middle_from_arr(arr, layer_df, target_mask, n_layers))
        n_target = int(target_mask.sum())
        n_words = len(target_mask)
        stratum_null = np.empty((n_perm, arr.shape[0]), dtype=float)
        for i in range(n_perm):
            sampled = np.zeros(n_words, dtype=bool)
            sampled[rng.choice(n_words, size=n_target, replace=False)] = True
            stratum_null[i, :] = late_minus_middle_from_arr(arr, layer_df, sampled, n_layers)
        null_parts.append(stratum_null)
    observed_unit = np.concatenate(observed_parts)
    null_units = np.concatenate(null_parts, axis=1)
    if statistic == "mean_late_minus_middle":
        return float(observed_unit.mean()), null_units.mean(axis=1)
    raise ValueError(f"Unknown statistic: {statistic}")


def self_candidate_permutation_null(
    word_df: pd.DataFrame,
    rng: np.random.Generator,
    n_perm: int,
) -> tuple[float, np.ndarray]:
    n_layers = int(word_df["layer_index"].max())
    arr, meta, layer_df, unit_meta = tensor_for_candidate_permutation(word_df, ["pair_id", "condition"])
    target_mask = meta["analysis_group"].eq("target").to_numpy()
    n_target = int(target_mask.sum())
    n_words = len(target_mask)
    bands = layer_bands(n_layers)
    layer_to_pos = dict(zip(layer_df["layer_index"].astype(int), layer_df["layer_pos"].astype(int)))
    middle_pos = [layer_to_pos[layer] for layer in bands["middle"] if layer in layer_to_pos]
    late_pos = [layer_to_pos[layer] for layer in bands["late"] if layer in layer_to_pos]
    def stat(mask: np.ndarray) -> float:
        margins = logsumexp_np(arr[:, :, mask], axis=2) - logsumexp_np(arr[:, :, ~mask], axis=2)
        delta = margins[:, late_pos].mean(axis=1) - margins[:, middle_pos].mean(axis=1)
        tmp = unit_meta.copy()
        tmp["delta"] = delta
        wide = tmp.pivot_table(index="pair_id", columns="condition", values="delta", aggfunc="mean")
        return float((wide["self"] - wide["object"]).mean())

    observed = stat(target_mask)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        sampled = np.zeros(n_words, dtype=bool)
        sampled[rng.choice(n_words, size=n_target, replace=False)] = True
        null[i] = stat(sampled)
    return observed, null


def lexical_control_sign_flip(rng: np.random.Generator, n_perm: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS_DIR.parent.parent / "outputs" / "control_lexical_layer_metrics.csv"
    metrics = pd.read_csv(path, low_memory=False)
    rows = []
    null_rows = []
    for keys, group in metrics.groupby(["model_name", "model_label", "condition_id", "candidate_set_version"], sort=False):
        model_name, model_label, condition_id, version = keys
        values = band_stat_from_margin_frame(
            group.rename(columns={"depth_minus_control_logit_margin": "margin"}),
            ["stem_id"],
        ).dropna().to_numpy()
        if len(values) < 2:
            continue
        null = sign_flip_null(values, rng, n_perm)
        rows.append(
            {
                "test_name": "lexical_control_stem_sign_flip",
                "probe_name": condition_id,
                "candidate_set_version": version,
                "model_name": model_name,
                "model_label": model_label,
                **permutation_summary(float(values.mean()), null),
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    "test_name": "lexical_control_stem_sign_flip",
                    "probe_name": condition_id,
                    "candidate_set_version": version,
                    "model_name": model_name,
                    "permutation_index": np.arange(n_perm),
                    "null_statistic": null,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(null_rows, ignore_index=True)


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(int(CONFIG["permutation_seed"]))
    n_perm = int(CONFIG["permutations"])
    metrics = aggregate_metrics()
    words = candidate_level_logits()
    results = []
    null_frames = []

    for model_id in CONFIG["models"]:
        model_metrics = metrics[metrics["model_id"].eq(model_id)]
        model_words = words[words["model_id"].eq(model_id)]
        model_label = label_for_model(model_id)

        values = self_pair_deltas(metrics, model_id).dropna().to_numpy()
        null = sign_flip_null(values, rng, n_perm)
        results.append(
            {
                "test_name": "self_object_pair_sign_flip",
                "probe_name": "self_object_advantage",
                "candidate_set_version": "full",
                "model_name": model_id,
                "model_label": model_label,
                **permutation_summary(float(values.mean()), null),
            }
        )
        null_frames.append(
            pd.DataFrame(
                {
                    "test_name": "self_object_pair_sign_flip",
                    "probe_name": "self_object_advantage",
                    "candidate_set_version": "full",
                    "model_name": model_id,
                    "permutation_index": np.arange(n_perm),
                    "null_statistic": null,
                }
            )
        )

        selected = model_words[model_words["probe_name"].eq("self_object")].copy()
        observed, null = self_candidate_permutation_null(selected, rng, n_perm)
        results.append(
            {
                "test_name": "self_object_candidate_label_permutation",
                "probe_name": "self_object_advantage",
                "candidate_set_version": "full",
                "model_name": model_id,
                "model_label": model_label,
                **permutation_summary(observed, null),
            }
        )
        null_frames.append(
            pd.DataFrame(
                {
                    "test_name": "self_object_candidate_label_permutation",
                    "probe_name": "self_object_advantage",
                    "candidate_set_version": "full",
                    "model_name": model_id,
                    "permutation_index": np.arange(n_perm),
                    "null_statistic": null,
                }
            )
        )

        for probe_name, strata in [
            ("reflective_practical", ["lexical_type"]),
            ("cova_verb_noun", ["lexical_type", "syntactic_frame"]),
        ]:
            selected = model_words[model_words["probe_name"].eq(probe_name)].copy()
            observed, null = candidate_permutation_null(
                selected,
                unit_cols=["prompt_id"],
                strata_cols=strata,
                statistic="mean_late_minus_middle",
                rng=rng,
                n_perm=n_perm,
            )
            results.append(
                {
                    "test_name": "candidate_label_permutation",
                    "probe_name": probe_name,
                    "candidate_set_version": "full",
                    "model_name": model_id,
                    "model_label": model_label,
                    **permutation_summary(observed, null),
                }
            )
            null_frames.append(
                pd.DataFrame(
                    {
                        "test_name": "candidate_label_permutation",
                        "probe_name": probe_name,
                        "candidate_set_version": "full",
                        "model_name": model_id,
                        "permutation_index": np.arange(n_perm),
                        "null_statistic": null,
                    }
                )
            )

    lexical_results, lexical_nulls = lexical_control_sign_flip(rng, n_perm)
    result_df = pd.concat([pd.DataFrame(results), lexical_results], ignore_index=True, sort=False)
    null_df = pd.concat(null_frames + [lexical_nulls], ignore_index=True, sort=False)

    adjusted = []
    group_cols = ["test_name", "probe_name", "candidate_set_version"]
    for _, group in result_df.groupby(group_cols, sort=False, dropna=False):
        group = group.copy()
        group["holm_p_two_sided_within_probe"] = holm_adjust(group["empirical_p_two_sided"])
        group["bh_q_two_sided_within_probe"] = bh_adjust(group["empirical_p_two_sided"])
        adjusted.append(group)
    result_df = pd.concat(adjusted, ignore_index=True, sort=False)

    save_table(result_df, RESULTS_DIR / "permutation_results")
    save_table(null_df, RESULTS_DIR / "permutation_nulls")

    report_lines = [
        "# Permutation Report",
        "",
        f"Monte Carlo permutations per test/model: {n_perm:,}.",
        "",
        "The primary self/object test includes a matched-pair sign-flip null and a candidate-label permutation null. Candidate permutations preserve part of speech or syntactic frame where applicable. Lexical-control rows use stem-level sign flips because the lexical-control output did not retain candidate-level logits for each matched subset.",
        "",
        "## Directional Summary",
        "",
    ]
    compact = result_df[
        [
            "test_name",
            "probe_name",
            "candidate_set_version",
            "model_label",
            "observed_statistic",
            "empirical_p_one_sided",
            "empirical_p_two_sided",
            "holm_p_two_sided_within_probe",
        ]
    ].copy()
    report_lines.append(compact.to_markdown(index=False, floatfmt=".4g"))
    (RESULTS_DIR / "permutation_results.csv").write_text(result_df.to_csv(index=False), encoding="utf-8")
    (RESULTS_DIR / "permutation_nulls.csv").write_text(null_df.to_csv(index=False), encoding="utf-8")
    (RESULTS_DIR.parent.parent / "reports" / "permutation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote permutation results for {len(result_df)} model/test rows.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-hoc null, Cova-dimension, and behavioral-alignment diagnostics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MIDDLE_MIN_RELATIVE = 0.25
MIDDLE_MAX_RELATIVE = 0.50
LATE_MIN_RELATIVE = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def band_layers(max_layer: int) -> dict[str, list[int]]:
    layers = list(range(max_layer + 1))
    middle = [
        layer
        for layer in layers
        if MIDDLE_MIN_RELATIVE <= layer / max_layer <= MIDDLE_MAX_RELATIVE and layer != 0
    ]
    late = [layer for layer in layers if LATE_MIN_RELATIVE < layer / max_layer < 1.0]
    if not middle:
        middle = [max(1, int(round(max_layer * 0.4)))]
    if not late:
        late = [max(1, max_layer - 1)]
    return {"middle": middle, "late": late, "final": [max_layer]}


def margin_for_group(group: pd.DataFrame, labels: dict[int, str]) -> float:
    values = group[["token_id", "word_logit"]].drop_duplicates("token_id")
    token_ids = values["token_id"].to_numpy(dtype=int)
    logits = values["word_logit"].to_numpy(dtype=float)
    assigned = np.array([labels.get(int(token_id), "") for token_id in token_ids], dtype=object)
    target = logits[assigned == "target"]
    contrast = logits[assigned == "contrast"]
    if len(target) == 0 or len(contrast) == 0:
        return np.nan
    return logsumexp(target) - logsumexp(contrast)


def late_minus_middle_from_layer_metrics(metrics: pd.DataFrame, value_col: str) -> float:
    max_layer = int(metrics["layer_index"].max())
    bands = band_layers(max_layer)
    by_unit = metrics.pivot_table(index="unit_id", columns="layer_index", values=value_col, aggfunc="mean")
    middle = by_unit[[layer for layer in bands["middle"] if layer in by_unit.columns]].mean(axis=1)
    late = by_unit[[layer for layer in bands["late"] if layer in by_unit.columns]].mean(axis=1)
    return float((late - middle).mean())


def shuffled_reflective_or_cova(
    word_metrics: pd.DataFrame,
    probe_name: str,
    rng: np.random.Generator,
    n_shuffles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = word_metrics[word_metrics["probe_name"].eq(probe_name)].copy()
    selected = selected[selected["stem_condition"].eq("all_stems") | selected["stem_condition"].eq("verb") | selected["stem_condition"].eq("noun")]
    rows = []
    null_rows = []
    for model_label, model_group in selected.groupby("model_label", sort=False):
        base = model_group.copy()
        if probe_name == "cova_verb_noun":
            base = base[base["stem_condition"].isin(["verb", "noun"])].copy()
            unit_cols = ["stem_id", "lexical_type"]
        else:
            unit_cols = ["stem_id"]
        token_table = (
            base[["syntactic_frame", "token_id", "analysis_group"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        real_labels = token_table.set_index("token_id")["analysis_group"].to_dict()
        real_unit = []
        for keys, group in base.groupby(unit_cols + ["layer_index"], sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            layer_index = keys[-1]
            unit_id = "::".join(str(x) for x in keys[:-1])
            real_unit.append(
                {
                    "unit_id": unit_id,
                    "layer_index": int(layer_index),
                    "margin": margin_for_group(group, real_labels),
                }
            )
        real_value = late_minus_middle_from_layer_metrics(pd.DataFrame(real_unit), "margin")
        for shuffle_index in range(n_shuffles):
            labels = {}
            for _frame, frame_tokens in token_table.groupby("syntactic_frame", sort=False):
                shuffled = frame_tokens["analysis_group"].to_numpy().copy()
                rng.shuffle(shuffled)
                labels.update(dict(zip(frame_tokens["token_id"], shuffled)))
            shuffled_unit = []
            for keys, group in base.groupby(unit_cols + ["layer_index"], sort=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                layer_index = keys[-1]
                unit_id = "::".join(str(x) for x in keys[:-1])
                shuffled_unit.append(
                    {
                        "unit_id": unit_id,
                        "layer_index": int(layer_index),
                        "margin": margin_for_group(group, labels),
                    }
                )
            null_value = late_minus_middle_from_layer_metrics(pd.DataFrame(shuffled_unit), "margin")
            null_rows.append(
                {
                    "model_label": model_label,
                    "probe_name": probe_name,
                    "shuffle_index": shuffle_index,
                    "null_late_minus_middle": null_value,
                    "real_late_minus_middle": real_value,
                }
            )
        null = pd.DataFrame([row for row in null_rows if row["model_label"] == model_label and row["probe_name"] == probe_name])
        rows.append(
            {
                "model_label": model_label,
                "probe_name": probe_name,
                "real_late_minus_middle": real_value,
                "null_mean": float(null["null_late_minus_middle"].mean()),
                "null_sd": float(null["null_late_minus_middle"].std(ddof=1)),
                "p_empirical_greater_equal": float((null["null_late_minus_middle"].ge(real_value).sum() + 1) / (len(null) + 1)),
                "real_exceeds_null_95th": bool(real_value > null["null_late_minus_middle"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def shuffled_self_advantage(
    word_metrics: pd.DataFrame,
    rng: np.random.Generator,
    n_shuffles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = word_metrics[word_metrics["probe_name"].eq("self_object")].copy()
    rows = []
    null_rows = []
    for model_label, model_group in selected.groupby("model_label", sort=False):
        token_table = model_group[["token_id", "analysis_group"]].drop_duplicates().reset_index(drop=True)
        real_labels = token_table.set_index("token_id")["analysis_group"].to_dict()

        def score_with_labels(labels: dict[int, str]) -> float:
            unit_rows = []
            for (pair_id, stem_condition, layer_index), group in model_group.groupby(
                ["pair_id", "stem_condition", "layer_index"], sort=False
            ):
                unit_rows.append(
                    {
                        "pair_id": pair_id,
                        "stem_condition": stem_condition,
                        "layer_index": int(layer_index),
                        "margin": margin_for_group(group, labels),
                    }
                )
            margins = pd.DataFrame(unit_rows)
            wide = margins.pivot_table(index=["pair_id", "layer_index"], columns="stem_condition", values="margin").reset_index()
            wide["unit_id"] = wide["pair_id"]
            wide["self_advantage"] = wide["self"] - wide["object"]
            return late_minus_middle_from_layer_metrics(wide[["unit_id", "layer_index", "self_advantage"]], "self_advantage")

        real_value = score_with_labels(real_labels)
        for shuffle_index in range(n_shuffles):
            shuffled = token_table["analysis_group"].to_numpy().copy()
            rng.shuffle(shuffled)
            labels = dict(zip(token_table["token_id"], shuffled))
            null_value = score_with_labels(labels)
            null_rows.append(
                {
                    "model_label": model_label,
                    "probe_name": "self_object_advantage",
                    "shuffle_index": shuffle_index,
                    "null_late_minus_middle": null_value,
                    "real_late_minus_middle": real_value,
                }
            )
        null = pd.DataFrame([row for row in null_rows if row["model_label"] == model_label and row["probe_name"] == "self_object_advantage"])
        rows.append(
            {
                "model_label": model_label,
                "probe_name": "self_object_advantage",
                "real_late_minus_middle": real_value,
                "null_mean": float(null["null_late_minus_middle"].mean()),
                "null_sd": float(null["null_late_minus_middle"].std(ddof=1)),
                "p_empirical_greater_equal": float((null["null_late_minus_middle"].ge(real_value).sum() + 1) / (len(null) + 1)),
                "real_exceeds_null_95th": bool(real_value > null["null_late_minus_middle"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def run_shuffled_nulls(word_metrics: pd.DataFrame, rng: np.random.Generator, n_shuffles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    nulls = []
    for probe_name in ["reflective_practical", "cova_verb_noun"]:
        summary, null = shuffled_reflective_or_cova(word_metrics, probe_name, rng, n_shuffles)
        summaries.append(summary)
        nulls.append(null)
    summary, null = shuffled_self_advantage(word_metrics, rng, n_shuffles)
    summaries.append(summary)
    nulls.append(null)
    return pd.concat(summaries, ignore_index=True), pd.concat(nulls, ignore_index=True)


def run_cova_dimension_robustness(word_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = word_metrics[word_metrics["probe_name"].eq("cova_verb_noun")].copy()
    controls = selected[selected["analysis_group"].eq("contrast")].copy()
    targets = selected[selected["analysis_group"].eq("target") & selected["cova_dimension"].notna()].copy()
    control_lse = (
        controls.groupby(["model_label", "stem_id", "lexical_type", "layer_index"], sort=False)["word_logit"]
        .apply(lambda values: logsumexp(values.to_numpy(dtype=float)))
        .reset_index(name="control_logsumexp")
    )
    target_lse = (
        targets.groupby(["model_label", "cova_dimension", "stem_id", "lexical_type", "layer_index"], sort=False)["word_logit"]
        .apply(lambda values: logsumexp(values.to_numpy(dtype=float)))
        .reset_index(name="target_logsumexp")
    )
    merged = target_lse.merge(
        control_lse,
        on=["model_label", "stem_id", "lexical_type", "layer_index"],
        how="inner",
    )
    merged["unit_id"] = merged["stem_id"].astype(str) + "::" + merged["lexical_type"].astype(str)
    merged["margin"] = merged["target_logsumexp"] - merged["control_logsumexp"]
    rows = []
    for (model_label, dimension), unit in merged.groupby(["model_label", "cova_dimension"], sort=False):
        max_layer = int(unit["layer_index"].max())
        bands = band_layers(max_layer)
        table = unit.pivot_table(index="unit_id", columns="layer_index", values="margin", aggfunc="mean")
        middle = table[[layer for layer in bands["middle"] if layer in table.columns]].mean(axis=1)
        late = table[[layer for layer in bands["late"] if layer in table.columns]].mean(axis=1)
        final = table[[layer for layer in bands["final"] if layer in table.columns]].mean(axis=1)
        rows.append(
            {
                "model_label": model_label,
                "cova_dimension": dimension,
                "n_units": int(len(table)),
                "middle_margin": float(middle.mean()),
                "late_margin": float(late.mean()),
                "final_margin": float(final.mean()),
                "late_minus_middle": float((late - middle).mean()),
                "late_above_final": bool(late.mean() > final.mean()),
                "positive_late_shift": bool((late - middle).mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def run_behavioral_alignment(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        (
            (summary["probe_name"].eq("reflective_practical") & summary["stem_condition"].eq("all_stems"))
            | (summary["probe_name"].eq("self_object_advantage") & summary["stem_condition"].eq("self_minus_object_stems"))
            | (summary["probe_name"].eq("cova_verb_noun") & summary["stem_condition"].eq("all_stems"))
        )
    ].copy()
    rows = []
    for probe_name, group in selected.groupby("probe_name", sort=False):
        corr = group["late_minus_middle"].corr(group["final_mean_margin"])
        rows.append(
            {
                "probe_name": probe_name,
                "n_models": int(group["model_label"].nunique()),
                "mean_late_minus_middle": float(group["late_minus_middle"].mean()),
                "mean_final_margin": float(group["final_mean_margin"].mean()),
                "models_positive_late_shift": int(group["late_minus_middle"].gt(0).sum()),
                "models_positive_final_margin": int(group["final_mean_margin"].gt(0).sum()),
                "correlation_late_shift_final_margin": float(corr) if not pd.isna(corr) else np.nan,
                "interpretation": "aligned" if group["final_mean_margin"].mean() > 0 else "late_internal_signal_exceeds_final_behavior",
            }
        )
    return pd.DataFrame(rows)


def plot_shuffled_summary(summary: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=180)
    labels = summary["probe_name"] + "\n" + summary["model_label"]
    x = np.arange(len(summary))
    ax.bar(x, summary["real_late_minus_middle"], color="#2563eb", label="real")
    ax.errorbar(x, summary["null_mean"], yerr=summary["null_sd"], fmt="o", color="#111827", label="shuffled mean +/- SD")
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Late-minus-middle margin")
    ax.set_title("Real candidate grouping vs shuffled-label null")
    ax.grid(axis="y", alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "posthoc_shuffled_label_null_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_cova_dimension(cova: pd.DataFrame, plot_dir: Path) -> None:
    table = cova.pivot_table(index="cova_dimension", columns="model_label", values="late_minus_middle", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10.5, max(5.2, 0.36 * len(table) + 1.8)), dpi=180)
    values = table.to_numpy(dtype=float)
    limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("Cova dimension robustness: late-minus-middle")
    ax.set_xlabel("Model")
    ax.set_ylabel("Cova dimension")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    fig.colorbar(image, ax=ax, label="Late-minus-middle margin")
    fig.tight_layout()
    fig.savefig(plot_dir / "posthoc_cova_dimension_robustness_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    word_metrics = pd.read_csv(output_dir / "cross_model_sentence_completion_word_metrics.csv")
    summary = pd.read_csv(output_dir / "cross_model_sentence_completion_summary_by_model.csv")
    rng = np.random.default_rng(args.seed)

    shuffled_summary, shuffled_null = run_shuffled_nulls(word_metrics, rng, args.n_shuffles)
    cova_dimension = run_cova_dimension_robustness(word_metrics)
    behavioral = run_behavioral_alignment(summary)

    shuffled_summary.to_csv(output_dir / "posthoc_shuffled_label_summary.csv", index=False)
    shuffled_null.to_csv(output_dir / "posthoc_shuffled_label_null_distribution.csv", index=False)
    cova_dimension.to_csv(output_dir / "posthoc_cova_dimension_robustness.csv", index=False)
    behavioral.to_csv(output_dir / "posthoc_behavioral_alignment_summary.csv", index=False)

    plot_shuffled_summary(shuffled_summary, plot_dir)
    plot_cova_dimension(cova_dimension, plot_dir)

    print("Shuffled-label summary:")
    print(shuffled_summary.to_string(index=False))
    print("\nBehavioral alignment:")
    print(behavioral.to_string(index=False))


if __name__ == "__main__":
    main()

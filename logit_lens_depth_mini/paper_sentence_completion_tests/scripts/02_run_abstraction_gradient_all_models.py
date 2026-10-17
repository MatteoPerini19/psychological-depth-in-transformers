#!/usr/bin/env python3
"""Cross-model abstraction-gradient Logit Lens probe for the paper package."""

from __future__ import annotations

import argparse
import gc
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from abstraction_gradient_engine import (
    GROUP_COLORS,
    GROUP_LABELS,
    build_metrics,
    choose_device,
    layer_logits_for_tokens,
    prepare_tokenizer_and_model,
    token_check_words,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_MODELS = [
    "gpt2",
    "distilgpt2",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m",
]
DEFAULT_STEMS_PATH = DATA_DIR / "abstraction_gradient_split_stems.csv"
DEFAULT_WORDS_PATH = DATA_DIR / "abstraction_gradient_split_word_groups.csv"

MIDDLE_MIN_RELATIVE = 0.25
MIDDLE_MAX_RELATIVE = 0.50
LATE_MIN_RELATIVE = 0.65


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def model_label(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.split("/")[-1])


def cleanup_model(model=None, tokenizer=None) -> None:
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def relative_band_layers(max_layer: int) -> dict[str, list[int]]:
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


def layer_region(layer_index: int | float, max_layer: int) -> str:
    if pd.isna(layer_index):
        return ""
    layer = int(layer_index)
    if layer == 0:
        return "embedding"
    if layer == max_layer:
        return "final"
    relative = layer / max_layer if max_layer else 0.0
    if MIDDLE_MIN_RELATIVE <= relative <= MIDDLE_MAX_RELATIVE:
        return "middle"
    if LATE_MIN_RELATIVE < relative < 1.0:
        return "late_intermediate"
    return "early"


def student_t_cdf(t_value: float, df: int) -> float:
    try:
        import mpmath as mp
    except ImportError:
        return float("nan")
    x = df / (df + t_value * t_value)
    integral = mp.betainc(df / 2, 0.5, 0, x, regularized=True)
    if t_value >= 0:
        return float(1 - 0.5 * integral)
    return float(0.5 * integral)


def paired_test(values: pd.Series) -> dict[str, float | int]:
    arr = values.dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n < 2:
        return {
            "n_stems": n,
            "mean": float(np.nanmean(arr)) if n else np.nan,
            "sd": np.nan,
            "t": np.nan,
            "df": n - 1,
            "p_two_sided": np.nan,
        }
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    t_value = mean / se if se > 0 else float("nan")
    df = n - 1
    p_value = 2 * (1 - student_t_cdf(abs(t_value), df)) if df > 0 else float("nan")
    return {"n_stems": n, "mean": mean, "sd": sd, "t": t_value, "df": df, "p_two_sided": p_value}


def summarize_group_layer(group_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        group_metrics.groupby(
            [
                "model_name",
                "model_label",
                "n_transformer_layers",
                "lexical_type",
                "abstraction_level",
                "word_group",
                "word_group_label",
                "layer_index",
                "relative_depth",
            ],
            sort=True,
        )
        .agg(
            mean_group_probability=("group_candidate_probability", "mean"),
            sem_group_probability=("group_candidate_probability", "sem"),
            mean_group_logsumexp=("group_logsumexp", "mean"),
            n_observations=("stem_id", "nunique"),
        )
        .reset_index()
    )


def summarize_score_layer(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(
            ["model_name", "model_label", "n_transformer_layers", "lexical_type", "layer_index", "relative_depth"],
            sort=True,
        )
        .agg(
            mean_expected_abstraction_level=("expected_abstraction_level", "mean"),
            sem_expected_abstraction_level=("expected_abstraction_level", "sem"),
            mean_high_abstraction_probability=("high_abstraction_probability", "mean"),
            mean_low_abstraction_probability=("low_abstraction_probability", "mean"),
            mean_high_minus_low_probability=("high_minus_low_probability", "mean"),
            mean_high_minus_low_logit_margin=("high_minus_low_logit_margin", "mean"),
            sem_high_minus_low_logit_margin=("high_minus_low_logit_margin", "sem"),
            n_stems=("stem_id", "nunique"),
        )
        .reset_index()
    )


def summarize_bands(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scores.groupby(["model_name", "model_label", "n_transformer_layers", "lexical_type"], sort=False):
        model_name, label, n_layers, lexical_type = keys
        bands = relative_band_layers(int(n_layers))
        pivot_expected = group.pivot_table(index="stem_id", columns="layer_index", values="expected_abstraction_level")
        pivot_margin = group.pivot_table(index="stem_id", columns="layer_index", values="high_minus_low_logit_margin")
        expected_middle = pivot_expected[[layer for layer in bands["middle"] if layer in pivot_expected.columns]].mean(axis=1)
        expected_late = pivot_expected[[layer for layer in bands["late"] if layer in pivot_expected.columns]].mean(axis=1)
        expected_final = pivot_expected[[layer for layer in bands["final"] if layer in pivot_expected.columns]].mean(axis=1)
        margin_middle = pivot_margin[[layer for layer in bands["middle"] if layer in pivot_margin.columns]].mean(axis=1)
        margin_late = pivot_margin[[layer for layer in bands["late"] if layer in pivot_margin.columns]].mean(axis=1)
        margin_final = pivot_margin[[layer for layer in bands["final"] if layer in pivot_margin.columns]].mean(axis=1)
        expected_curve = group.groupby("layer_index", sort=True)["expected_abstraction_level"].mean()
        margin_curve = group.groupby("layer_index", sort=True)["high_minus_low_logit_margin"].mean()
        expected_peak_layer = int(expected_curve.idxmax())
        margin_peak_layer = int(margin_curve.idxmax())
        expected_test = paired_test(expected_late - expected_middle)
        margin_test = paired_test(margin_late - margin_middle)
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "probe_name": "abstraction_gradient",
                "probe_label": "Abstraction gradient",
                "lexical_type": lexical_type,
                "n_transformer_layers": int(n_layers),
                "n_stems": int(len(pivot_expected)),
                "middle_layers": " ".join(str(layer) for layer in bands["middle"]),
                "late_layers": " ".join(str(layer) for layer in bands["late"]),
                "final_layer": int(bands["final"][0]),
                "expected_middle_mean": float(expected_middle.mean()),
                "expected_late_mean": float(expected_late.mean()),
                "expected_final_mean": float(expected_final.mean()),
                "expected_late_minus_middle": float((expected_late - expected_middle).mean()),
                "expected_late_minus_middle_t": expected_test["t"],
                "expected_late_minus_middle_df": expected_test["df"],
                "expected_late_minus_middle_p_two_sided": expected_test["p_two_sided"],
                "expected_peak_layer": expected_peak_layer,
                "expected_peak_relative_depth": float(expected_peak_layer / int(n_layers)),
                "expected_peak_region": layer_region(expected_peak_layer, int(n_layers)),
                "margin_middle_mean": float(margin_middle.mean()),
                "margin_late_mean": float(margin_late.mean()),
                "margin_final_mean": float(margin_final.mean()),
                "margin_late_minus_middle": float((margin_late - margin_middle).mean()),
                "margin_late_minus_middle_t": margin_test["t"],
                "margin_late_minus_middle_df": margin_test["df"],
                "margin_late_minus_middle_p_two_sided": margin_test["p_two_sided"],
                "margin_peak_layer": margin_peak_layer,
                "margin_peak_relative_depth": float(margin_peak_layer / int(n_layers)),
                "margin_peak_region": layer_region(margin_peak_layer, int(n_layers)),
            }
        )
    return pd.DataFrame(rows)


def plot_probability_by_type(group_layer: pd.DataFrame, plot_dir: Path) -> None:
    for lexical_type, selected in group_layer.groupby("lexical_type", sort=True):
        fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), dpi=180, sharex=True, sharey=True)
        axes = axes.flatten()
        for ax, (label, model_group) in zip(axes, selected.groupby("model_label", sort=False)):
            for word_group, sub in model_group.groupby("word_group", sort=False):
                sub = sub.sort_values("relative_depth")
                ax.plot(
                    sub["relative_depth"],
                    sub["mean_group_probability"],
                    marker="o",
                    linewidth=1.9,
                    color=GROUP_COLORS[word_group],
                    label=GROUP_LABELS[word_group],
                )
            ax.axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.05)
            ax.set_title(label)
            ax.grid(True, alpha=0.24)
        for ax in axes[len(selected["model_label"].unique()) :]:
            ax.axis("off")
        for ax in axes[::3]:
            ax.set_ylabel("Candidate-set probability")
        for ax in axes[-3:]:
            ax.set_xlabel("Relative depth")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
        fig.suptitle(f"{lexical_type.title()} abstraction-gradient probability mass", y=0.99)
        fig.tight_layout(rect=(0, 0.055, 1, 0.96))
        fig.savefig(plot_dir / f"abstraction_gradient_cross_model_{lexical_type}_probability.png", bbox_inches="tight")
        plt.close(fig)


def plot_score_by_type(score_layer: pd.DataFrame, plot_dir: Path) -> None:
    for lexical_type, selected in score_layer.groupby("lexical_type", sort=True):
        fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), dpi=180, sharex=True)
        for label, model_group in selected.groupby("model_label", sort=False):
            model_group = model_group.sort_values("relative_depth")
            axes[0].plot(
                model_group["relative_depth"],
                model_group["mean_expected_abstraction_level"],
                marker="o",
                linewidth=2.0,
                label=label,
            )
            axes[1].plot(
                model_group["relative_depth"],
                model_group["mean_high_minus_low_logit_margin"],
                marker="o",
                linewidth=2.0,
                label=label,
            )
        axes[0].axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.05)
        axes[0].set_title("Expected abstraction score")
        axes[0].set_ylabel("Expected level")
        axes[0].set_ylim(1, 4)
        axes[1].axhline(0, color="#444444", linewidth=1)
        axes[1].axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.05)
        axes[1].set_title("High-abstraction minus low-abstraction margin")
        axes[1].set_ylabel("Logsumexp margin")
        for ax in axes:
            ax.set_xlabel("Relative depth")
            ax.grid(True, alpha=0.24)
            ax.legend(fontsize=8)
        fig.suptitle(f"{lexical_type.title()} abstraction-gradient cross-model scores", y=1.02)
        fig.tight_layout()
        fig.savefig(plot_dir / f"abstraction_gradient_cross_model_{lexical_type}_scores.png", bbox_inches="tight")
        plt.close(fig)


def plot_late_shift_bars(band_summary: pd.DataFrame, plot_dir: Path) -> None:
    for metric, title, output_name in [
        ("expected_late_minus_middle", "Expected abstraction late-minus-middle shift", "expected_late_shift"),
        ("margin_late_minus_middle", "High-minus-low late-minus-middle margin shift", "margin_late_shift"),
    ]:
        pivot = band_summary.pivot_table(index="lexical_type", columns="model_label", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(11.8, 5.5), dpi=180)
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Lexical type")
        ax.set_ylabel("Late minus middle")
        ax.grid(True, axis="y", alpha=0.24)
        ax.legend(title="Model", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"abstraction_gradient_cross_model_{output_name}.png", bbox_inches="tight")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    stems = pd.read_csv(DEFAULT_STEMS_PATH)
    words = pd.read_csv(DEFAULT_WORDS_PATH)
    device = choose_device(args.device)

    all_group_metrics = []
    all_scores = []
    all_word_metrics = []
    all_token_checks = []
    errors = []

    for model_name in args.models:
        tokenizer = None
        model = None
        print(f"\nLoading {model_name} on {device}...")
        try:
            tokenizer, model = prepare_tokenizer_and_model(model_name, device)
            token_check = token_check_words(words, tokenizer)
            token_check["model_name"] = model_name
            token_check["model_label"] = model_label(model_name)
            all_token_checks.append(token_check)
            usable_words = token_check[token_check["is_usable_primary"]].reset_index(drop=True)
            token_ids = sorted(int(token_id) for token_id in usable_words["token_id"].unique())
            print(
                f"Scoring {len(stems)} stems and {len(usable_words)} usable candidates "
                f"for {model_name}..."
            )
            logits = layer_logits_for_tokens(
                model=model,
                tokenizer=tokenizer,
                stems=stems["stem"].tolist(),
                token_ids=token_ids,
                device=device,
                batch_size=args.batch_size,
            )
            group_metrics, scores, word_metrics = build_metrics(stems, usable_words, logits, token_ids, model_name)
            final_layer = int(scores["layer_index"].max())
            for frame in [group_metrics, scores, word_metrics]:
                frame["model_label"] = model_label(model_name)
                frame["n_transformer_layers"] = final_layer
            all_group_metrics.append(group_metrics)
            all_scores.append(scores)
            all_word_metrics.append(word_metrics)
        except Exception as exc:
            errors.append({"model_name": model_name, "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")
        finally:
            cleanup_model(model, tokenizer)

    if all_token_checks:
        pd.concat(all_token_checks, ignore_index=True).to_csv(
            output_dir / "abstraction_gradient_cross_model_token_check.csv",
            index=False,
        )
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "abstraction_gradient_cross_model_errors.csv", index=False)
    if not all_scores:
        raise RuntimeError("No abstraction-gradient model runs completed successfully.")

    group_metrics = pd.concat(all_group_metrics, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    word_metrics = pd.concat(all_word_metrics, ignore_index=True)
    group_layer = summarize_group_layer(group_metrics)
    score_layer = summarize_score_layer(scores)
    band_summary = summarize_bands(scores)

    group_metrics.to_csv(output_dir / "abstraction_gradient_cross_model_group_metrics.csv", index=False)
    scores.to_csv(output_dir / "abstraction_gradient_cross_model_stem_scores.csv", index=False)
    word_metrics.to_csv(output_dir / "abstraction_gradient_cross_model_word_metrics.csv", index=False)
    group_layer.to_csv(output_dir / "abstraction_gradient_cross_model_summary_by_group_layer.csv", index=False)
    score_layer.to_csv(output_dir / "abstraction_gradient_cross_model_summary_by_layer.csv", index=False)
    band_summary.to_csv(output_dir / "abstraction_gradient_cross_model_band_summary.csv", index=False)

    plot_probability_by_type(group_layer, plot_dir)
    plot_score_by_type(score_layer, plot_dir)
    plot_late_shift_bars(band_summary, plot_dir)

    print("\nDone.")
    print("Completed abstraction-gradient models:")
    print(", ".join(band_summary["model_label"].drop_duplicates().tolist()))
    print(f"Wrote CSVs to {output_dir}")
    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()

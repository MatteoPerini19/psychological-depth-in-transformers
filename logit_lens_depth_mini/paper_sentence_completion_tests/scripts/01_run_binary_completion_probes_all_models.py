#!/usr/bin/env python3
"""Cross-model Logit Lens replication for sentence-completion depth probes."""

from __future__ import annotations

import argparse
import gc
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

MIDDLE_MIN_RELATIVE = 0.25
MIDDLE_MAX_RELATIVE = 0.50
LATE_MIN_RELATIVE = 0.65


@dataclass(frozen=True)
class ProbeSpec:
    probe_name: str
    probe_label: str
    stems_path: Path
    words_path: Path
    match_syntactic_frame: bool = False


PROBES = [
    ProbeSpec(
        probe_name="reflective_practical",
        probe_label="Reflective vs practical",
        stems_path=DATA_DIR / "simple_word_group_stems.csv",
        words_path=DATA_DIR / "simple_word_groups.csv",
    ),
    ProbeSpec(
        probe_name="self_object",
        probe_label="Inner-life vs object",
        stems_path=DATA_DIR / "self_object_stems.csv",
        words_path=DATA_DIR / "self_object_word_groups.csv",
    ),
    ProbeSpec(
        probe_name="cova_verb_noun",
        probe_label="Cova verb/noun depth vs control",
        stems_path=DATA_DIR / "cova_verb_noun_stems.csv",
        words_path=DATA_DIR / "cova_depth_completion_words_single_token.csv",
        match_syntactic_frame=True,
    ),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=[probe.probe_name for probe in PROBES],
        default=[probe.probe_name for probe in PROBES],
    )
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def model_label(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.split("/")[-1])


def normalize_for_lm_head(model, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm(hidden)
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    return hidden


def output_logits_for_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings")
    return output_embeddings(normalize_for_lm_head(model, hidden))


def prepare_tokenizer_and_model(model_name: str, device: torch.device, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return tokenizer, model


def token_check(tokenizer, token_text: str) -> tuple[bool, int | float, str]:
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    token_id: int | float = int(token_ids[0]) if len(token_ids) == 1 else np.nan
    return len(token_ids) == 1, token_id, " ".join(str(token_id) for token_id in token_ids)


def load_stems(spec: ProbeSpec) -> pd.DataFrame:
    stems = pd.read_csv(spec.stems_path)
    if "stem_id" not in stems.columns or "stem" not in stems.columns:
        raise ValueError(f"{spec.stems_path} must contain stem_id and stem columns.")

    stems = stems.copy()
    if spec.probe_name == "reflective_practical":
        stems["stem_condition"] = "all_stems"
        stems["syntactic_frame"] = "verb_after_to"
        stems["lexical_type"] = "verb"
    elif spec.probe_name == "self_object":
        stems["stem_condition"] = stems["stem_family"]
        stems["syntactic_frame"] = "noun_after_possessive_or_article"
        stems["lexical_type"] = "noun"
    elif spec.probe_name == "cova_verb_noun":
        stems["stem_condition"] = stems["lexical_type"]
    else:
        stems["stem_condition"] = "all_stems"

    for column in ["name", "pronoun", "pair_id", "stem_template", "stem_family"]:
        if column not in stems.columns:
            stems[column] = ""
    return stems


def load_reflective_practical_candidates(spec: ProbeSpec, tokenizer, model_name: str) -> pd.DataFrame:
    words = pd.read_csv(spec.words_path)
    rows = []
    for _, row in words.iterrows():
        token_text = " " + str(row["word"])
        is_single, token_id, token_ids = token_check(tokenizer, token_text)
        rows.append(
            {
                "model_name": model_name,
                "probe_name": spec.probe_name,
                "word": row["word"],
                "token_text": token_text,
                "token_id": token_id,
                "token_ids": token_ids,
                "is_single_token": is_single,
                "source_word_group": row["word_group"],
                "analysis_group": "target" if row["word_group"] == "deep" else "contrast",
                "analysis_group_label": "reflective" if row["word_group"] == "deep" else "practical",
                "role": row.get("role", ""),
                "syntactic_frame": "verb_after_to",
                "lexical_type": "verb",
                "cova_dimension": "",
                "source_basis": "hand_picked_minimal_stem_words",
            }
        )
    return pd.DataFrame(rows)


def load_self_object_candidates(spec: ProbeSpec, tokenizer, model_name: str) -> pd.DataFrame:
    words = pd.read_csv(spec.words_path)
    rows = []
    for _, row in words.iterrows():
        token_text = " " + str(row["word"])
        is_single, token_id, token_ids = token_check(tokenizer, token_text)
        rows.append(
            {
                "model_name": model_name,
                "probe_name": spec.probe_name,
                "word": row["word"],
                "token_text": token_text,
                "token_id": token_id,
                "token_ids": token_ids,
                "is_single_token": is_single,
                "source_word_group": row["word_group"],
                "analysis_group": "target" if row["word_group"] == "inner" else "contrast",
                "analysis_group_label": "inner_life" if row["word_group"] == "inner" else "object_logistic",
                "role": row.get("role", ""),
                "syntactic_frame": "noun_after_possessive_or_article",
                "lexical_type": "noun",
                "cova_dimension": "",
                "source_basis": "hand_picked_self_object_words",
            }
        )
    return pd.DataFrame(rows)


def load_cova_candidates(spec: ProbeSpec, tokenizer, model_name: str) -> pd.DataFrame:
    words = pd.read_csv(spec.words_path)
    rows = []
    usable_source = words[
        words["lexical_type"].isin(["verb", "noun"])
        & words["word_group"].isin(["depth", "surface", "neutral_control"])
    ].copy()
    for _, row in usable_source.iterrows():
        token_text = str(row["token_text"])
        if not token_text.startswith(" "):
            token_text = " " + str(row["word"])
        is_single, token_id, token_ids = token_check(tokenizer, token_text)
        source_group = row["word_group"]
        rows.append(
            {
                "model_name": model_name,
                "probe_name": spec.probe_name,
                "word": row["word"],
                "token_text": token_text,
                "token_id": token_id,
                "token_ids": token_ids,
                "is_single_token": is_single,
                "source_word_group": source_group,
                "analysis_group": "target" if source_group == "depth" else "contrast",
                "analysis_group_label": "cova_depth_positive" if source_group == "depth" else "practical_control",
                "role": row.get("polarity", ""),
                "syntactic_frame": row["syntactic_frame"],
                "lexical_type": row["lexical_type"],
                "cova_dimension": row.get("cova_dimension", ""),
                "source_basis": row.get("source_basis", ""),
            }
        )
    return pd.DataFrame(rows)


def load_candidates(spec: ProbeSpec, tokenizer, model_name: str) -> pd.DataFrame:
    if spec.probe_name == "reflective_practical":
        candidates = load_reflective_practical_candidates(spec, tokenizer, model_name)
    elif spec.probe_name == "self_object":
        candidates = load_self_object_candidates(spec, tokenizer, model_name)
    elif spec.probe_name == "cova_verb_noun":
        candidates = load_cova_candidates(spec, tokenizer, model_name)
    else:
        raise ValueError(f"Unknown probe: {spec.probe_name}")

    candidates["is_usable"] = candidates["is_single_token"]
    return candidates


def usable_candidates_for_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    usable = candidates[candidates["is_usable"]].copy()
    duplicate_keys = [
        "probe_name",
        "syntactic_frame",
        "analysis_group",
        "token_id",
    ]
    usable = usable.drop_duplicates(subset=duplicate_keys).reset_index(drop=True)
    conflicts = usable.groupby(["syntactic_frame", "token_id"])["analysis_group"].nunique()
    conflicting_ids = conflicts[conflicts > 1].reset_index()[["syntactic_frame", "token_id"]]
    if len(conflicting_ids):
        conflict_pairs = set(map(tuple, conflicting_ids.to_numpy()))
        usable = usable[
            ~usable.apply(lambda row: (row["syntactic_frame"], row["token_id"]) in conflict_pairs, axis=1)
        ].copy()
    return usable.reset_index(drop=True)


def layer_logits_for_tokens(
    model,
    tokenizer,
    stems: list[str],
    token_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(stems), batch_size):
        batch_stems = stems[start : start + batch_size]
        encoded = tokenizer(batch_stems, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        final_positions = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(len(batch_stems), device=device)
        per_layer = []
        for hidden in outputs.hidden_states:
            last_hidden = hidden[batch_index, final_positions, :]
            logits = output_logits_for_hidden(model, last_hidden)
            per_layer.append(logits[:, token_ids].detach().cpu().numpy())
        batches.append(np.stack(per_layer, axis=1))
    return np.concatenate(batches, axis=0)


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_probe_metrics(
    spec: ProbeSpec,
    stems: pd.DataFrame,
    candidates: pd.DataFrame,
    logits: np.ndarray,
    token_ids: list[int],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_to_col = {token_id: col for col, token_id in enumerate(token_ids)}
    max_layer_index = logits.shape[1] - 1
    metric_rows = []
    word_rows = []

    for stem_index, stem in stems.iterrows():
        if spec.match_syntactic_frame:
            stem_candidates = candidates[candidates["syntactic_frame"].eq(stem["syntactic_frame"])].reset_index(drop=True)
        else:
            stem_candidates = candidates.reset_index(drop=True)

        target_mask = stem_candidates["analysis_group"].to_numpy() == "target"
        contrast_mask = stem_candidates["analysis_group"].to_numpy() == "contrast"
        if not target_mask.any() or not contrast_mask.any():
            continue

        candidate_cols = [id_to_col[int(token_id)] for token_id in stem_candidates["token_id"]]
        for layer_index in range(logits.shape[1]):
            values = logits[stem_index, layer_index, candidate_cols]
            probabilities = softmax(values)
            target_lse = logsumexp(values[target_mask])
            contrast_lse = logsumexp(values[contrast_mask])
            target_prob = float(probabilities[target_mask].sum())
            contrast_prob = float(probabilities[contrast_mask].sum())
            relative_depth = layer_index / max_layer_index if max_layer_index else 0.0
            base = {
                "model_name": model_name,
                "model_label": model_label(model_name),
                "n_transformer_layers": max_layer_index,
                "probe_name": spec.probe_name,
                "probe_label": spec.probe_label,
                "stem_id": stem["stem_id"],
                "stem": stem["stem"],
                "stem_template": stem["stem_template"],
                "stem_condition": stem["stem_condition"],
                "stem_family": stem["stem_family"],
                "pair_id": stem["pair_id"],
                "syntactic_frame": stem["syntactic_frame"],
                "lexical_type": stem["lexical_type"],
                "layer_index": layer_index,
                "relative_depth": relative_depth,
                "layer_name": "embedding" if layer_index == 0 else f"block_{layer_index}",
                "target_group_label": stem_candidates.loc[target_mask, "analysis_group_label"].iloc[0],
                "contrast_group_label": stem_candidates.loc[contrast_mask, "analysis_group_label"].iloc[0],
                "n_target_words": int(target_mask.sum()),
                "n_contrast_words": int(contrast_mask.sum()),
            }
            metric_rows.append(
                {
                    **base,
                    "target_logsumexp": target_lse,
                    "contrast_logsumexp": contrast_lse,
                    "target_minus_contrast_logit_margin": target_lse - contrast_lse,
                    "target_candidate_probability": target_prob,
                    "contrast_candidate_probability": contrast_prob,
                    "target_minus_contrast_probability": target_prob - contrast_prob,
                }
            )
            for word_index, word in stem_candidates.iterrows():
                word_rows.append(
                    {
                        **base,
                        "word": word["word"],
                        "token_text": word["token_text"],
                        "token_id": int(word["token_id"]),
                        "source_word_group": word["source_word_group"],
                        "analysis_group": word["analysis_group"],
                        "analysis_group_label": word["analysis_group_label"],
                        "role": word["role"],
                        "cova_dimension": word["cova_dimension"],
                        "source_basis": word["source_basis"],
                        "word_logit": float(values[word_index]),
                        "candidate_set_probability": float(probabilities[word_index]),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(word_rows)


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


def summarize_by_layer(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            [
                "model_name",
                "model_label",
                "n_transformer_layers",
                "probe_name",
                "probe_label",
                "stem_condition",
                "layer_index",
                "relative_depth",
            ],
            sort=False,
        )
        .agg(
            mean_target_probability=("target_candidate_probability", "mean"),
            mean_contrast_probability=("contrast_candidate_probability", "mean"),
            mean_probability_gap=("target_minus_contrast_probability", "mean"),
            mean_logit_margin=("target_minus_contrast_logit_margin", "mean"),
            sd_logit_margin=("target_minus_contrast_logit_margin", "std"),
            sem_logit_margin=("target_minus_contrast_logit_margin", "sem"),
            n_observations=("target_minus_contrast_logit_margin", "size"),
            mean_n_target_words=("n_target_words", "mean"),
            mean_n_contrast_words=("n_contrast_words", "mean"),
        )
        .reset_index()
    )


def make_all_condition_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model_name", "probe_name"]
    for (_model_name, _probe_name), group in metrics.groupby(group_cols, sort=False):
        if group["stem_condition"].nunique() <= 1:
            continue
        copied = group.copy()
        copied["stem_condition"] = "all_stems"
        rows.append(copied)
    if rows:
        return pd.concat([metrics] + rows, ignore_index=True)
    return metrics


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
    values = values.dropna().to_numpy(dtype=float)
    n = int(len(values))
    if n < 2:
        return {"n_units": n, "mean": float(np.nanmean(values)) if n else np.nan, "sd": np.nan, "t": np.nan, "df": n - 1, "p_two_sided": np.nan}
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    t_value = mean / se if se > 0 else float("nan")
    df = n - 1
    p_value = 2 * (1 - student_t_cdf(abs(t_value), df)) if df > 0 else float("nan")
    return {"n_units": n, "mean": mean, "sd": sd, "t": t_value, "df": df, "p_two_sided": p_value}


def summarize_bands(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unit_cols = ["stem_id"]
    for keys, group in metrics.groupby(
        ["model_name", "model_label", "n_transformer_layers", "probe_name", "probe_label", "stem_condition"],
        sort=False,
    ):
        model_name, label, n_layers, probe_name, probe_label, condition = keys
        bands = relative_band_layers(int(n_layers))
        pivot = group.pivot_table(
            index=unit_cols,
            columns="layer_index",
            values="target_minus_contrast_logit_margin",
            aggfunc="mean",
        )
        prob_pivot = group.pivot_table(
            index=unit_cols,
            columns="layer_index",
            values="target_candidate_probability",
            aggfunc="mean",
        )
        contrast_prob_pivot = group.pivot_table(
            index=unit_cols,
            columns="layer_index",
            values="contrast_candidate_probability",
            aggfunc="mean",
        )
        middle = pivot[[layer for layer in bands["middle"] if layer in pivot.columns]].mean(axis=1)
        late = pivot[[layer for layer in bands["late"] if layer in pivot.columns]].mean(axis=1)
        final = pivot[[layer for layer in bands["final"] if layer in pivot.columns]].mean(axis=1)
        late_target_prob = prob_pivot[[layer for layer in bands["late"] if layer in prob_pivot.columns]].mean(axis=1)
        late_contrast_prob = contrast_prob_pivot[[layer for layer in bands["late"] if layer in contrast_prob_pivot.columns]].mean(axis=1)
        by_layer = (
            group.groupby("layer_index", sort=True)["target_minus_contrast_logit_margin"]
            .mean()
            .reset_index()
        )
        peak = by_layer.loc[by_layer["target_minus_contrast_logit_margin"].idxmax()]
        peak_layer = int(peak["layer_index"])
        transformer_layers = by_layer[by_layer["layer_index"].gt(0)]
        transformer_peak = transformer_layers.loc[
            transformer_layers["target_minus_contrast_logit_margin"].idxmax()
        ]
        transformer_peak_layer = int(transformer_peak["layer_index"])
        test = paired_test(late - middle)
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "probe_name": probe_name,
                "probe_label": probe_label,
                "stem_condition": condition,
                "n_transformer_layers": int(n_layers),
                "n_stems": int(len(pivot)),
                "middle_layers": " ".join(str(layer) for layer in bands["middle"]),
                "late_layers": " ".join(str(layer) for layer in bands["late"]),
                "final_layer": int(bands["final"][0]),
                "middle_mean_margin": float(middle.mean()),
                "middle_sd_margin": float(middle.std(ddof=1)),
                "late_mean_margin": float(late.mean()),
                "late_sd_margin": float(late.std(ddof=1)),
                "late_minus_middle": float((late - middle).mean()),
                "late_minus_middle_sd": float((late - middle).std(ddof=1)),
                "late_minus_middle_t": test["t"],
                "late_minus_middle_df": test["df"],
                "late_minus_middle_p_two_sided": test["p_two_sided"],
                "final_mean_margin": float(final.mean()),
                "late_minus_final": float((late - final).mean()),
                "late_target_probability": float(late_target_prob.mean()),
                "late_contrast_probability": float(late_contrast_prob.mean()),
                "peak_layer": peak_layer,
                "peak_relative_depth": float(peak_layer / int(n_layers)),
                "strongest_layer_region": layer_region(peak_layer, int(n_layers)),
                "strongest_layer_is_late_intermediate_or_final": layer_region(peak_layer, int(n_layers))
                in {"late_intermediate", "final"},
                "peak_mean_margin": float(peak["target_minus_contrast_logit_margin"]),
                "peak_transformer_layer": transformer_peak_layer,
                "peak_transformer_relative_depth": float(transformer_peak_layer / int(n_layers)),
                "strongest_transformer_layer_region": layer_region(transformer_peak_layer, int(n_layers)),
                "strongest_transformer_layer_is_late_intermediate_or_final": layer_region(
                    transformer_peak_layer, int(n_layers)
                )
                in {"late_intermediate", "final"},
                "peak_transformer_mean_margin": float(transformer_peak["target_minus_contrast_logit_margin"]),
                "mean_n_target_words": float(group["n_target_words"].mean()),
                "mean_n_contrast_words": float(group["n_contrast_words"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_self_object_advantage(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["probe_name"].eq("self_object")]
    if selected.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in selected.groupby(["model_name", "model_label", "n_transformer_layers", "pair_id"], sort=False):
        model_name, label, n_layers, pair_id = keys
        bands = relative_band_layers(int(n_layers))
        family_layer = (
            group.groupby(["stem_condition", "layer_index"])["target_minus_contrast_logit_margin"]
            .mean()
            .reset_index()
        )
        row = {
            "model_name": model_name,
            "model_label": label,
            "n_transformer_layers": int(n_layers),
            "probe_name": "self_object_advantage",
            "probe_label": "Self-directed advantage",
            "stem_condition": "self_minus_object_stems",
            "pair_id": pair_id,
        }
        for family in ["self", "object"]:
            family_values = family_layer[family_layer["stem_condition"].eq(family)].set_index("layer_index")[
                "target_minus_contrast_logit_margin"
            ]
            for band_name, layers in bands.items():
                available = [layer for layer in layers if layer in family_values.index]
                row[f"{family}_{band_name}_margin"] = float(family_values.loc[available].mean())
        row["middle_margin"] = row["self_middle_margin"] - row["object_middle_margin"]
        row["late_margin"] = row["self_late_margin"] - row["object_late_margin"]
        row["final_margin"] = row["self_final_margin"] - row["object_final_margin"]
        row["late_minus_middle"] = row["late_margin"] - row["middle_margin"]
        rows.append(row)

    pair_rows = pd.DataFrame(rows)
    summary_rows = []
    if pair_rows.empty:
        return pd.DataFrame()
    for keys, group in pair_rows.groupby(["model_name", "model_label", "n_transformer_layers"], sort=False):
        model_name, label, n_layers = keys
        test = paired_test(group["late_minus_middle"])
        model_metrics = selected[
            (selected["model_name"].eq(model_name))
        ]
        layer_advantage = (
            model_metrics.groupby(["pair_id", "stem_condition", "layer_index"], sort=True)[
                "target_minus_contrast_logit_margin"
            ]
            .mean()
            .reset_index()
            .pivot_table(
                index=["pair_id", "layer_index"],
                columns="stem_condition",
                values="target_minus_contrast_logit_margin",
                aggfunc="mean",
            )
            .reset_index()
        )
        if {"self", "object"}.issubset(layer_advantage.columns):
            layer_advantage["self_minus_object"] = layer_advantage["self"] - layer_advantage["object"]
            peak_by_layer = (
                layer_advantage.groupby("layer_index", sort=True)["self_minus_object"]
                .mean()
                .reset_index()
            )
            peak = peak_by_layer.loc[peak_by_layer["self_minus_object"].idxmax()]
            peak_layer = int(peak["layer_index"])
            peak_margin = float(peak["self_minus_object"])
            transformer_peak_by_layer = peak_by_layer[peak_by_layer["layer_index"].gt(0)]
            transformer_peak = transformer_peak_by_layer.loc[transformer_peak_by_layer["self_minus_object"].idxmax()]
            transformer_peak_layer = int(transformer_peak["layer_index"])
            transformer_peak_margin = float(transformer_peak["self_minus_object"])
        else:
            peak_layer = np.nan
            peak_margin = np.nan
            transformer_peak_layer = np.nan
            transformer_peak_margin = np.nan
        summary_rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "probe_name": "self_object_advantage",
                "probe_label": "Self-directed advantage",
                "stem_condition": "self_minus_object_stems",
                "n_transformer_layers": int(n_layers),
                "n_stems": int(len(group)),
                "middle_layers": "",
                "late_layers": "",
                "final_layer": int(n_layers),
                "middle_mean_margin": float(group["middle_margin"].mean()),
                "middle_sd_margin": float(group["middle_margin"].std(ddof=1)),
                "late_mean_margin": float(group["late_margin"].mean()),
                "late_sd_margin": float(group["late_margin"].std(ddof=1)),
                "late_minus_middle": float(group["late_minus_middle"].mean()),
                "late_minus_middle_sd": float(group["late_minus_middle"].std(ddof=1)),
                "late_minus_middle_t": test["t"],
                "late_minus_middle_df": test["df"],
                "late_minus_middle_p_two_sided": test["p_two_sided"],
                "final_mean_margin": float(group["final_margin"].mean()),
                "late_minus_final": float((group["late_margin"] - group["final_margin"]).mean()),
                "late_target_probability": np.nan,
                "late_contrast_probability": np.nan,
                "peak_layer": peak_layer,
                "peak_relative_depth": float(peak_layer / int(n_layers)) if not pd.isna(peak_layer) else np.nan,
                "strongest_layer_region": layer_region(peak_layer, int(n_layers)) if not pd.isna(peak_layer) else "",
                "strongest_layer_is_late_intermediate_or_final": layer_region(peak_layer, int(n_layers))
                in {"late_intermediate", "final"}
                if not pd.isna(peak_layer)
                else False,
                "peak_mean_margin": peak_margin,
                "peak_transformer_layer": transformer_peak_layer,
                "peak_transformer_relative_depth": float(transformer_peak_layer / int(n_layers))
                if not pd.isna(transformer_peak_layer)
                else np.nan,
                "strongest_transformer_layer_region": layer_region(transformer_peak_layer, int(n_layers))
                if not pd.isna(transformer_peak_layer)
                else "",
                "strongest_transformer_layer_is_late_intermediate_or_final": layer_region(
                    transformer_peak_layer, int(n_layers)
                )
                in {"late_intermediate", "final"}
                if not pd.isna(transformer_peak_layer)
                else False,
                "peak_transformer_mean_margin": transformer_peak_margin,
                "mean_n_target_words": np.nan,
                "mean_n_contrast_words": np.nan,
            }
        )
    return pd.DataFrame(summary_rows)


def plot_combined_margin(by_layer: pd.DataFrame, output_path: Path) -> None:
    focus = by_layer[
        (
            (by_layer["probe_name"].eq("reflective_practical") & by_layer["stem_condition"].eq("all_stems"))
            | (by_layer["probe_name"].eq("self_object") & by_layer["stem_condition"].isin(["self", "object"]))
            | (by_layer["probe_name"].eq("cova_verb_noun") & by_layer["stem_condition"].eq("all_stems"))
        )
    ].copy()
    focus["panel"] = np.select(
        [
            focus["probe_name"].eq("reflective_practical"),
            focus["probe_name"].eq("self_object") & focus["stem_condition"].eq("self"),
            focus["probe_name"].eq("self_object") & focus["stem_condition"].eq("object"),
            focus["probe_name"].eq("cova_verb_noun"),
        ],
        [
            "Reflective - practical",
            "Self stems: inner - object",
            "Object stems: inner - object",
            "Cova verbs/nouns - control",
        ],
        default=focus["probe_label"],
    )
    panels = [
        "Reflective - practical",
        "Self stems: inner - object",
        "Object stems: inner - object",
        "Cova verbs/nouns - control",
    ]
    panels = [panel for panel in panels if panel in set(focus["panel"])]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.0), dpi=180, sharex=True)
    axes = axes.flatten()
    for ax, panel in zip(axes, panels):
        group = focus[focus["panel"].eq(panel)]
        for label, model_group in group.groupby("model_label", sort=False):
            ax.plot(
                model_group["relative_depth"],
                model_group["mean_logit_margin"],
                marker="o",
                linewidth=2.1,
                label=label,
            )
        ax.axhline(0, color="#444444", linewidth=1)
        ax.axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.06)
        ax.set_title(panel)
        ax.set_ylabel("Mean logsumexp margin")
        ax.grid(True, alpha=0.24)
        ax.legend(fontsize=8)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    for ax in axes[-2:]:
        ax.set_xlabel("Relative depth")
    fig.suptitle("Cross-model sentence-completion Logit Lens margins", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_combined_probabilities(by_layer: pd.DataFrame, output_path: Path) -> None:
    focus = by_layer[
        (
            (by_layer["probe_name"].eq("reflective_practical") & by_layer["stem_condition"].eq("all_stems"))
            | (by_layer["probe_name"].eq("self_object") & by_layer["stem_condition"].eq("self"))
            | (by_layer["probe_name"].eq("cova_verb_noun") & by_layer["stem_condition"].eq("all_stems"))
        )
    ].copy()
    focus["panel"] = np.select(
        [
            focus["probe_name"].eq("reflective_practical"),
            focus["probe_name"].eq("self_object"),
            focus["probe_name"].eq("cova_verb_noun"),
        ],
        [
            "Reflective/practical",
            "Self stems: inner/object",
            "Cova verb/noun",
        ],
        default=focus["probe_label"],
    )
    panels = ["Reflective/practical", "Self stems: inner/object", "Cova verb/noun"]
    panels = [panel for panel in panels if panel in set(focus["panel"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, max(4.0, 3.2 * len(panels))), dpi=180, sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        group = focus[focus["panel"].eq(panel)]
        for label, model_group in group.groupby("model_label", sort=False):
            ax.plot(
                model_group["relative_depth"],
                model_group["mean_target_probability"],
                marker="o",
                linewidth=2.0,
                label=f"{label}: target",
            )
            ax.plot(
                model_group["relative_depth"],
                model_group["mean_contrast_probability"],
                marker="x",
                linestyle="--",
                linewidth=1.6,
                label=f"{label}: contrast",
            )
        ax.axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.05)
        ax.set_title(panel)
        ax.set_ylabel("Candidate-set probability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.24)
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Relative depth")
    fig.suptitle("Cross-model candidate-set probabilities", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_late_shift(summary: pd.DataFrame, output_path: Path) -> None:
    focus = summary[
        (
            (summary["probe_name"].eq("reflective_practical") & summary["stem_condition"].eq("all_stems"))
            | (summary["probe_name"].eq("self_object") & summary["stem_condition"].isin(["self", "object"]))
            | (summary["probe_name"].eq("self_object_advantage"))
            | (summary["probe_name"].eq("cova_verb_noun") & summary["stem_condition"].eq("all_stems"))
        )
    ].copy()
    focus["label"] = np.select(
        [
            focus["probe_name"].eq("reflective_practical"),
            focus["probe_name"].eq("self_object") & focus["stem_condition"].eq("self"),
            focus["probe_name"].eq("self_object") & focus["stem_condition"].eq("object"),
            focus["probe_name"].eq("self_object_advantage"),
            focus["probe_name"].eq("cova_verb_noun"),
        ],
        ["reflective", "self stems", "object stems", "self advantage", "Cova verb/noun"],
        default=focus["probe_name"] + ":" + focus["stem_condition"],
    )
    pivot = focus.pivot_table(index="label", columns="model_label", values="late_minus_middle", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11.8, 6.0), dpi=180)
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("Late-minus-middle shift by model and sentence-completion probe")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Late minus middle logit margin")
    ax.grid(True, axis="y", alpha=0.24)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = [probe for probe in PROBES if probe.probe_name in set(args.probes)]
    device = choose_device(args.device)

    all_metrics = []
    all_word_metrics = []
    all_token_checks = []
    errors = []

    for model_name in args.models:
        print(f"\nLoading {model_name} on {device}...")
        try:
            tokenizer, model = prepare_tokenizer_and_model(model_name, device, args.local_files_only)
        except Exception as exc:
            errors.append({"model_name": model_name, "probe_name": "model_load", "error": repr(exc)})
            print(f"ERROR loading {model_name}: {exc}")
            continue

        try:
            for spec in selected_specs:
                print(f"Scoring {model_name}: {spec.probe_label}...")
                stems = load_stems(spec)
                token_checks = load_candidates(spec, tokenizer, model_name)
                all_token_checks.append(token_checks)
                candidates = usable_candidates_for_metrics(token_checks)
                if candidates.empty:
                    raise ValueError(f"No usable single-token candidates for {spec.probe_name}.")

                group_counts = candidates.groupby("analysis_group").size().to_dict()
                if group_counts.get("target", 0) == 0 or group_counts.get("contrast", 0) == 0:
                    raise ValueError(f"Missing target or contrast candidates for {spec.probe_name}: {group_counts}")
                token_ids = sorted(int(token_id) for token_id in candidates["token_id"].unique())
                logits = layer_logits_for_tokens(
                    model=model,
                    tokenizer=tokenizer,
                    stems=stems["stem"].tolist(),
                    token_ids=token_ids,
                    device=device,
                    batch_size=args.batch_size,
                )
                metrics, word_metrics = build_probe_metrics(
                    spec=spec,
                    stems=stems,
                    candidates=candidates,
                    logits=logits,
                    token_ids=token_ids,
                    model_name=model_name,
                )
                if metrics.empty:
                    raise ValueError(f"No metric rows generated for {spec.probe_name}.")
                all_metrics.append(metrics)
                all_word_metrics.append(word_metrics)
        except Exception as exc:
            errors.append({"model_name": model_name, "probe_name": "probe_loop", "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")
        finally:
            cleanup_model(model)
            del tokenizer

    if all_token_checks:
        pd.concat(all_token_checks, ignore_index=True).to_csv(
            output_dir / "cross_model_sentence_completion_token_check.csv",
            index=False,
        )
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "cross_model_sentence_completion_errors.csv", index=False)
    if not all_metrics:
        raise RuntimeError("No model/probe combinations completed successfully.")

    metrics = pd.concat(all_metrics, ignore_index=True)
    word_metrics = pd.concat(all_word_metrics, ignore_index=True)
    metrics_with_all = make_all_condition_metrics(metrics)
    by_layer = summarize_by_layer(metrics_with_all)
    summary = summarize_bands(metrics_with_all)
    self_advantage = summarize_self_object_advantage(metrics)
    if not self_advantage.empty:
        summary = pd.concat([summary, self_advantage], ignore_index=True, sort=False)

    metrics.to_csv(output_dir / "cross_model_sentence_completion_layer_metrics.csv", index=False)
    word_metrics.to_csv(output_dir / "cross_model_sentence_completion_word_metrics.csv", index=False)
    by_layer.to_csv(output_dir / "cross_model_sentence_completion_summary_by_layer.csv", index=False)
    summary.to_csv(output_dir / "cross_model_sentence_completion_summary_by_model.csv", index=False)

    plot_combined_margin(by_layer, plot_dir / "cross_model_sentence_completion_margin_relative_depth.png")
    plot_combined_probabilities(by_layer, plot_dir / "cross_model_sentence_completion_probability_relative_depth.png")
    plot_late_shift(summary, plot_dir / "cross_model_sentence_completion_late_shift_bars.png")

    print("\nDone.")
    completed = (
        metrics[["model_label", "probe_name"]]
        .drop_duplicates()
        .assign(pair=lambda frame: frame["model_label"] + ":" + frame["probe_name"])
    )
    print("Completed model/probe pairs:")
    print("\n".join(completed["pair"].tolist()))
    if errors:
        print(f"Errors were written to {output_dir / 'cross_model_sentence_completion_errors.csv'}")
    print(f"Wrote CSVs to {output_dir}")
    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()

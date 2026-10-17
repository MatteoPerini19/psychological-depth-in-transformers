#!/usr/bin/env python3
"""All-model lexical-control and motivational-specificity tests."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-matched-pairs", type=int, default=6)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


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


def prepare_tokenizer_and_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return tokenizer, model


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


def token_info(tokenizer, word: str) -> tuple[bool, str, int | float, str]:
    token_text = " " + str(word)
    token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    token_id = int(token_ids[0]) if len(token_ids) == 1 else np.nan
    return len(token_ids) == 1, token_text, token_id, " ".join(str(token_id) for token_id in token_ids)


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def layer_logits_for_tokens(
    model,
    tokenizer,
    prompts: list[str],
    token_ids: list[int],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batches = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start : start + batch_size], return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        final_positions = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(encoded["input_ids"].shape[0], device=device)
        per_layer = []
        for hidden in outputs.hidden_states:
            final_hidden = hidden[batch_index, final_positions, :]
            logits = output_logits_for_hidden(model, final_hidden)
            per_layer.append(logits[:, token_ids].detach().cpu().numpy())
        batches.append(np.stack(per_layer, axis=1))
    return np.concatenate(batches, axis=0)


def final_logits_for_tokens(model, tokenizer, prompts: list[str], token_ids: list[int], device: torch.device) -> np.ndarray:
    logits = layer_logits_for_tokens(model, tokenizer, prompts, token_ids, device, batch_size=len(prompts))
    return logits[:, -1, :]


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


def abstractness_proxy(word: str, group: str, lexical_type: str) -> float:
    inner = {"life", "values", "identity", "future", "feelings", "past", "purpose", "beliefs", "choices", "relationships"}
    objects = {"phone", "schedule", "order", "room", "report", "email", "bill", "car", "list", "box"}
    reflective = {"understand", "reflect", "reconsider", "question", "change", "learn", "forgive", "heal", "grow", "remember"}
    practical = {"buy", "check", "call", "text", "watch", "eat", "shop", "clean", "pay", "order"}
    word = str(word).lower()
    if word in objects:
        return 0.10
    if word in practical:
        return 0.25
    if word in inner:
        return 0.88
    if word in reflective:
        return 0.72
    if group == "depth_related":
        return 0.75 if lexical_type == "noun" else 0.68
    return 0.30


def build_lexical_candidates_and_stems(tokenizer, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = []
    stems = []
    token_rows = []

    simple_words = pd.read_csv(DATA_DIR / "simple_word_groups.csv")
    for _, row in simple_words.iterrows():
        is_single, token_text, token_id, token_ids = token_info(tokenizer, row["word"])
        token_rows.append({"model_name": model_name, "condition_id": "reflective_vs_practical_verbs", **row.to_dict(), "token_ids": token_ids, "is_single_token": is_single})
        if is_single:
            candidates.append(
                {
                    "condition_id": "reflective_vs_practical_verbs",
                    "condition_label": "Reflective vs practical verbs",
                    "word": row["word"],
                    "token_text": token_text,
                    "token_id": int(token_id),
                    "token_ids": token_ids,
                    "candidate_group": "depth_related" if row["word_group"] == "deep" else "control",
                    "lexical_type": "verb",
                    "syntactic_frame": "verb_after_to",
                    "source": "simple_word_groups",
                }
            )
    simple_stems = pd.read_csv(DATA_DIR / "simple_word_group_stems.csv")
    simple_stems["condition_id"] = "reflective_vs_practical_verbs"
    simple_stems["condition_label"] = "Reflective vs practical verbs"
    simple_stems["lexical_type"] = "verb"
    simple_stems["syntactic_frame"] = "verb_after_to"
    simple_stems["stem_family"] = "minimal_verb"
    stems.append(simple_stems)

    self_words = pd.read_csv(DATA_DIR / "self_object_word_groups.csv")
    for _, row in self_words.iterrows():
        is_single, token_text, token_id, token_ids = token_info(tokenizer, row["word"])
        token_rows.append({"model_name": model_name, "condition_id": "inner_life_vs_object_nouns", **row.to_dict(), "token_ids": token_ids, "is_single_token": is_single})
        if is_single:
            candidates.append(
                {
                    "condition_id": "inner_life_vs_object_nouns",
                    "condition_label": "Inner-life vs object nouns",
                    "word": row["word"],
                    "token_text": token_text,
                    "token_id": int(token_id),
                    "token_ids": token_ids,
                    "candidate_group": "depth_related" if row["word_group"] == "inner" else "control",
                    "lexical_type": "noun",
                    "syntactic_frame": "noun_after_possessive_or_article",
                    "source": "self_object_word_groups",
                }
            )
    self_stems = pd.read_csv(DATA_DIR / "self_object_stems.csv")
    self_stems = self_stems[self_stems["stem_family"].eq("self")].copy()
    self_stems["condition_id"] = "inner_life_vs_object_nouns"
    self_stems["condition_label"] = "Inner-life vs object nouns"
    self_stems["lexical_type"] = "noun"
    self_stems["syntactic_frame"] = "noun_after_possessive_or_article"
    stems.append(self_stems)

    cova_words = pd.read_csv(DATA_DIR / "cova_depth_completion_words_single_token.csv")
    cova_words = cova_words[
        cova_words["lexical_type"].isin(["verb", "noun"])
        & cova_words["word_group"].isin(["depth", "surface", "neutral_control"])
    ].copy()
    cova_words = cova_words.drop_duplicates(["word", "lexical_type", "syntactic_frame", "word_group"])
    for _, row in cova_words.iterrows():
        is_single, token_text, token_id, token_ids = token_info(tokenizer, row["word"])
        token_rows.append({"model_name": model_name, "condition_id": "cova_verb_noun_depth_vs_control", "word": row["word"], "token_ids": token_ids, "is_single_token": is_single})
        if is_single:
            candidates.append(
                {
                    "condition_id": "cova_verb_noun_depth_vs_control",
                    "condition_label": "Cova-derived verbs/nouns",
                    "word": row["word"],
                    "token_text": token_text,
                    "token_id": int(token_id),
                    "token_ids": token_ids,
                    "candidate_group": "depth_related" if row["word_group"] == "depth" else "control",
                    "lexical_type": row["lexical_type"],
                    "syntactic_frame": row["syntactic_frame"],
                    "source": "cova_depth_completion_words_single_token",
                }
            )
    cova_stems = pd.read_csv(DATA_DIR / "cova_verb_noun_stems.csv")
    cova_stems["condition_id"] = "cova_verb_noun_depth_vs_control"
    cova_stems["condition_label"] = "Cova-derived verbs/nouns"
    stems.append(cova_stems)

    candidates_df = pd.DataFrame(candidates).drop_duplicates(
        ["condition_id", "word", "token_id", "candidate_group", "lexical_type", "syntactic_frame"]
    )
    candidates_df["token_length_chars"] = candidates_df["word"].astype(str).str.len()
    candidates_df["token_id_frequency_proxy"] = -np.log(candidates_df["token_id"].astype(float) + 1.0)
    candidates_df["abstractness_proxy"] = candidates_df.apply(
        lambda row: abstractness_proxy(row["word"], row["candidate_group"], row["lexical_type"]),
        axis=1,
    )
    return candidates_df.reset_index(drop=True), pd.concat(stems, ignore_index=True, sort=False), pd.DataFrame(token_rows)


def add_neutral_frequency_proxy(candidates: pd.DataFrame, model, tokenizer, device: torch.device) -> pd.DataFrame:
    neutral_stems = {
        "verb_after_to": ["Someone wants to", "Someone needs to", "A person tries to"],
        "noun_after_possessive": ["Someone thinks about his", "Someone considers her", "A person examines his"],
        "noun_after_possessive_or_article": ["Someone thinks about his", "Someone considers her", "A person examines the"],
    }
    rows = []
    for frame, group in candidates.groupby("syntactic_frame", sort=False):
        prompts = neutral_stems.get(frame, ["Someone mentions"])
        token_ids = [int(token_id) for token_id in group["token_id"]]
        logits = final_logits_for_tokens(model, tokenizer, prompts, token_ids, device)
        mean_logits = logits.mean(axis=0)
        for (_, row), mean_logit in zip(group.iterrows(), mean_logits):
            rows.append({**row.to_dict(), "neutral_logit_frequency_proxy": float(mean_logit)})
    return pd.DataFrame(rows).reset_index(drop=True)


def match_candidates(candidates: pd.DataFrame, max_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidates.copy()
    candidates["selected_for_matched_subset"] = False
    candidates["matched_pair_id"] = ""
    candidates["match_distance"] = np.nan
    match_rows = []
    features = ["neutral_logit_frequency_proxy", "token_id_frequency_proxy", "token_length_chars", "abstractness_proxy"]
    for keys, group in candidates.groupby(["condition_id", "lexical_type", "syntactic_frame"], sort=False):
        condition_id, lexical_type, syntactic_frame = keys
        depth = group[group["candidate_group"].eq("depth_related")]
        control = group[group["candidate_group"].eq("control")]
        if depth.empty or control.empty:
            continue
        standardized = group[features].astype(float).copy()
        for feature in features:
            sd = standardized[feature].std(ddof=0)
            standardized[feature] = 0.0 if sd == 0 or np.isnan(sd) else (standardized[feature] - standardized[feature].mean()) / sd
        standardized.index = group.index
        pair_rows = []
        for depth_index, depth_row in depth.iterrows():
            for control_index, control_row in control.iterrows():
                diff = standardized.loc[depth_index, features] - standardized.loc[control_index, features]
                distance = float(np.sqrt(np.square(diff).sum()))
                pair_rows.append(
                    {
                        "condition_id": condition_id,
                        "lexical_type": lexical_type,
                        "syntactic_frame": syntactic_frame,
                        "depth_index": int(depth_index),
                        "control_index": int(control_index),
                        "depth_word": depth_row["word"],
                        "control_word": control_row["word"],
                        "match_distance": distance,
                    }
                )
        pair_table = pd.DataFrame(pair_rows).sort_values("match_distance")
        used_depth: set[int] = set()
        used_control: set[int] = set()
        pair_count = 0
        for _, pair in pair_table.iterrows():
            depth_index = int(pair["depth_index"])
            control_index = int(pair["control_index"])
            if depth_index in used_depth or control_index in used_control:
                continue
            pair_count += 1
            pair_id = f"{condition_id}_{lexical_type}_{pair_count:02d}"
            used_depth.add(depth_index)
            used_control.add(control_index)
            candidates.loc[[depth_index, control_index], "selected_for_matched_subset"] = True
            candidates.loc[[depth_index, control_index], "matched_pair_id"] = pair_id
            candidates.loc[[depth_index, control_index], "match_distance"] = pair["match_distance"]
            match_rows.append({**pair.to_dict(), "matched_pair_id": pair_id})
            if pair_count >= min(max_pairs, len(depth), len(control)):
                break
    return candidates, pd.DataFrame(match_rows)


def score_lexical_layers(
    model_name: str,
    stems: pd.DataFrame,
    candidates: pd.DataFrame,
    logits: np.ndarray,
    token_ids: list[int],
) -> pd.DataFrame:
    id_to_col = {token_id: col for col, token_id in enumerate(token_ids)}
    rows = []
    final_layer = logits.shape[1] - 1
    for stem_index, stem in stems.reset_index(drop=True).iterrows():
        base_candidates = candidates[
            candidates["condition_id"].eq(stem["condition_id"])
            & candidates["lexical_type"].eq(stem["lexical_type"])
            & candidates["syntactic_frame"].eq(stem["syntactic_frame"])
        ]
        for version, selected in [
            ("all_candidates", base_candidates),
            ("lexically_matched", base_candidates[base_candidates["selected_for_matched_subset"]]),
        ]:
            depth = selected[selected["candidate_group"].eq("depth_related")]
            control = selected[selected["candidate_group"].eq("control")]
            if depth.empty or control.empty:
                continue
            selected = pd.concat([depth, control], ignore_index=True)
            cols = [id_to_col[int(token_id)] for token_id in selected["token_id"]]
            depth_mask = selected["candidate_group"].eq("depth_related").to_numpy()
            control_mask = selected["candidate_group"].eq("control").to_numpy()
            for layer_index in range(logits.shape[1]):
                values = logits[stem_index, layer_index, cols]
                probs = softmax(values)
                rows.append(
                    {
                        "model_name": model_name,
                        "model_label": model_label(model_name),
                        "n_transformer_layers": final_layer,
                        "condition_id": stem["condition_id"],
                        "condition_label": stem["condition_label"],
                        "candidate_set_version": version,
                        "stem_id": stem["stem_id"],
                        "stem": stem["stem"],
                        "stem_template": stem.get("stem_template", ""),
                        "stem_family": stem.get("stem_family", ""),
                        "lexical_type": stem["lexical_type"],
                        "syntactic_frame": stem["syntactic_frame"],
                        "layer_index": layer_index,
                        "relative_depth": layer_index / final_layer if final_layer else 0,
                        "depth_minus_control_logit_margin": logsumexp(values[depth_mask]) - logsumexp(values[control_mask]),
                        "depth_candidate_probability": float(probs[depth_mask].sum()),
                        "control_candidate_probability": float(probs[control_mask].sum()),
                        "n_depth_candidates": int(depth_mask.sum()),
                        "n_control_candidates": int(control_mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_lexical(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["model_name", "model_label", "condition_id", "condition_label", "candidate_set_version"], sort=False):
        model_name, label, condition_id, condition_label, version = keys
        max_layer = int(group["n_transformer_layers"].iloc[0])
        bands = band_layers(max_layer)
        table = group.pivot_table(index="stem_id", columns="layer_index", values="depth_minus_control_logit_margin", aggfunc="mean")
        middle = table[[layer for layer in bands["middle"] if layer in table.columns]].mean(axis=1)
        late = table[[layer for layer in bands["late"] if layer in table.columns]].mean(axis=1)
        final = table[[layer for layer in bands["final"] if layer in table.columns]].mean(axis=1)
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "condition_id": condition_id,
                "condition_label": condition_label,
                "candidate_set_version": version,
                "n_stems": int(len(table)),
                "middle_margin": float(middle.mean()),
                "late_margin": float(late.mean()),
                "final_margin": float(final.mean()),
                "late_minus_middle": float((late - middle).mean()),
                "survives_matching": bool(version == "lexically_matched" and (late - middle).mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def build_intentional_materials(tokenizer, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stems = pd.read_csv(DATA_DIR / "intentional_mechanical_stems.csv")
    words = pd.read_csv(DATA_DIR / "simple_word_groups.csv")
    candidates = []
    token_rows = []
    for _, row in words.iterrows():
        is_single, token_text, token_id, token_ids = token_info(tokenizer, row["word"])
        token_rows.append({"model_name": model_name, **row.to_dict(), "token_ids": token_ids, "is_single_token": is_single})
        if is_single:
            candidates.append(
                {
                    "word": row["word"],
                    "token_text": token_text,
                    "token_id": int(token_id),
                    "candidate_group": "depth_related" if row["word_group"] == "deep" else "control",
                    "word_group": row["word_group"],
                }
            )
    return stems, pd.DataFrame(candidates), pd.DataFrame(token_rows)


def score_intentional_layers(model_name: str, stems: pd.DataFrame, candidates: pd.DataFrame, logits: np.ndarray, token_ids: list[int]) -> pd.DataFrame:
    id_to_col = {token_id: col for col, token_id in enumerate(token_ids)}
    cols = [id_to_col[int(token_id)] for token_id in candidates["token_id"]]
    depth_mask = candidates["candidate_group"].eq("depth_related").to_numpy()
    control_mask = candidates["candidate_group"].eq("control").to_numpy()
    rows = []
    final_layer = logits.shape[1] - 1
    for stem_index, stem in stems.iterrows():
        for layer_index in range(logits.shape[1]):
            values = logits[stem_index, layer_index, cols]
            probs = softmax(values)
            rows.append(
                {
                    "model_name": model_name,
                    "model_label": model_label(model_name),
                    "n_transformer_layers": final_layer,
                    "stem_id": stem["stem_id"],
                    "stem_family": stem["stem_family"],
                    "stem_template": stem["stem_template"],
                    "stem": stem["stem"],
                    "layer_index": layer_index,
                    "relative_depth": layer_index / final_layer if final_layer else 0,
                    "depth_minus_control_logit_margin": logsumexp(values[depth_mask]) - logsumexp(values[control_mask]),
                    "depth_candidate_probability": float(probs[depth_mask].sum()),
                    "control_candidate_probability": float(probs[control_mask].sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_intentional(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["model_name", "model_label", "stem_family"], sort=False):
        model_name, label, family = keys
        max_layer = int(group["n_transformer_layers"].iloc[0])
        bands = band_layers(max_layer)
        table = group.pivot_table(index="stem_id", columns="layer_index", values="depth_minus_control_logit_margin", aggfunc="mean")
        middle = table[[layer for layer in bands["middle"] if layer in table.columns]].mean(axis=1)
        late = table[[layer for layer in bands["late"] if layer in table.columns]].mean(axis=1)
        final = table[[layer for layer in bands["final"] if layer in table.columns]].mean(axis=1)
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "stem_family": family,
                "n_stems": int(len(table)),
                "middle_margin": float(middle.mean()),
                "late_margin": float(late.mean()),
                "final_margin": float(final.mean()),
                "late_minus_middle": float((late - middle).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    advantage_rows = []
    for (model_name, label), group in summary.groupby(["model_name", "model_label"], sort=False):
        wide = group.set_index("stem_family")
        if {"intentional", "mechanical"}.issubset(wide.index):
            advantage_rows.append(
                {
                    "model_name": model_name,
                    "model_label": label,
                    "stem_family": "intentional_minus_mechanical",
                    "n_stems": int(wide.loc["intentional", "n_stems"] + wide.loc["mechanical", "n_stems"]),
                    "middle_margin": float(wide.loc["intentional", "middle_margin"] - wide.loc["mechanical", "middle_margin"]),
                    "late_margin": float(wide.loc["intentional", "late_margin"] - wide.loc["mechanical", "late_margin"]),
                    "final_margin": float(wide.loc["intentional", "final_margin"] - wide.loc["mechanical", "final_margin"]),
                    "late_minus_middle": float(wide.loc["intentional", "late_minus_middle"] - wide.loc["mechanical", "late_minus_middle"]),
                }
            )
    return pd.concat([summary, pd.DataFrame(advantage_rows)], ignore_index=True)


def plot_lexical_summary(summary: pd.DataFrame, plot_dir: Path) -> None:
    selected = summary[summary["candidate_set_version"].eq("lexically_matched")].copy()
    selected["label"] = selected["condition_id"].str.replace("_", " ") + "\n" + selected["model_label"]
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=180)
    colors = ["#2563eb" if value > 0 else "#c2410c" for value in selected["late_minus_middle"]]
    ax.bar(np.arange(len(selected)), selected["late_minus_middle"], color=colors)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_xticks(np.arange(len(selected)))
    ax.set_xticklabels(selected["label"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Late-minus-middle margin after lexical matching")
    ax.set_title("Lexical-control robustness across models")
    ax.grid(axis="y", alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_dir / "control_lexical_matched_late_shift.png", bbox_inches="tight")
    plt.close(fig)


def plot_intentional_summary(summary: pd.DataFrame, plot_dir: Path) -> None:
    selected = summary[summary["stem_family"].eq("intentional_minus_mechanical")].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=180)
    colors = ["#2563eb" if value > 0 else "#c2410c" for value in selected["late_minus_middle"]]
    ax.bar(selected["model_label"], selected["late_minus_middle"], color=colors)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("Motivational specificity: intentional minus mechanical late shift")
    ax.set_ylabel("Late-minus-middle advantage")
    ax.grid(axis="y", alpha=0.24)
    fig.tight_layout()
    fig.savefig(plot_dir / "control_intentional_mechanical_late_shift.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    all_lexical_metrics = []
    all_lexical_summary = []
    all_matches = []
    all_lexical_candidates = []
    all_intentional_metrics = []
    all_intentional_summary = []
    all_token_checks = []
    errors = []

    for model_name in args.models:
        print(f"\nLoading {model_name} on {device}...")
        tokenizer = None
        model = None
        try:
            tokenizer, model = prepare_tokenizer_and_model(model_name, device)

            candidates, stems, token_check = build_lexical_candidates_and_stems(tokenizer, model_name)
            candidates = add_neutral_frequency_proxy(candidates, model, tokenizer, device)
            candidates, matches = match_candidates(candidates, args.max_matched_pairs)
            candidates["model_name"] = model_name
            candidates["model_label"] = model_label(model_name)
            matches["model_name"] = model_name
            matches["model_label"] = model_label(model_name)
            token_check["model_label"] = model_label(model_name)
            all_token_checks.append(token_check)
            token_ids = sorted(int(token_id) for token_id in candidates["token_id"].unique())
            print(f"Scoring lexical-control conditions for {model_name}...")
            logits = layer_logits_for_tokens(model, tokenizer, stems["stem"].tolist(), token_ids, device, args.batch_size)
            lexical_metrics = score_lexical_layers(model_name, stems, candidates, logits, token_ids)
            lexical_summary = summarize_lexical(lexical_metrics)
            all_lexical_metrics.append(lexical_metrics)
            all_lexical_summary.append(lexical_summary)
            all_matches.append(matches)
            all_lexical_candidates.append(candidates)

            intentional_stems, intentional_candidates, intentional_tokens = build_intentional_materials(tokenizer, model_name)
            intentional_tokens["model_label"] = model_label(model_name)
            all_token_checks.append(intentional_tokens)
            token_ids = sorted(int(token_id) for token_id in intentional_candidates["token_id"].unique())
            print(f"Scoring intentional/mechanical stems for {model_name}...")
            logits = layer_logits_for_tokens(model, tokenizer, intentional_stems["stem"].tolist(), token_ids, device, args.batch_size)
            intentional_metrics = score_intentional_layers(model_name, intentional_stems, intentional_candidates, logits, token_ids)
            intentional_summary = summarize_intentional(intentional_metrics)
            all_intentional_metrics.append(intentional_metrics)
            all_intentional_summary.append(intentional_summary)
        except Exception as exc:
            errors.append({"model_name": model_name, "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")
        finally:
            cleanup_model(model, tokenizer)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "control_specificity_errors.csv", index=False)
    if all_token_checks:
        pd.concat(all_token_checks, ignore_index=True, sort=False).to_csv(output_dir / "control_specificity_token_check.csv", index=False)
    if not all_lexical_metrics or not all_intentional_metrics:
        raise RuntimeError("No control/specificity runs completed.")

    lexical_metrics = pd.concat(all_lexical_metrics, ignore_index=True)
    lexical_summary = pd.concat(all_lexical_summary, ignore_index=True)
    matches = pd.concat(all_matches, ignore_index=True, sort=False)
    candidates = pd.concat(all_lexical_candidates, ignore_index=True, sort=False)
    intentional_metrics = pd.concat(all_intentional_metrics, ignore_index=True)
    intentional_summary = pd.concat(all_intentional_summary, ignore_index=True)

    lexical_metrics.to_csv(output_dir / "control_lexical_layer_metrics.csv", index=False)
    lexical_summary.to_csv(output_dir / "control_lexical_summary_by_model.csv", index=False)
    matches.to_csv(output_dir / "control_lexical_matched_pairs.csv", index=False)
    candidates.to_csv(output_dir / "control_lexical_candidates_with_matching.csv", index=False)
    intentional_metrics.to_csv(output_dir / "control_intentional_mechanical_layer_metrics.csv", index=False)
    intentional_summary.to_csv(output_dir / "control_intentional_mechanical_summary_by_model.csv", index=False)

    plot_lexical_summary(lexical_summary, plot_dir)
    plot_intentional_summary(intentional_summary, plot_dir)

    print("\nLexical-control summary:")
    print(lexical_summary.to_string(index=False))
    print("\nIntentional/mechanical summary:")
    print(intentional_summary.to_string(index=False))


if __name__ == "__main__":
    main()

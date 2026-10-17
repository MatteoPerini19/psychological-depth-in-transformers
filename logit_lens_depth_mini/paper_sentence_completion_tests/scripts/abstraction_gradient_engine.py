#!/usr/bin/env python3
"""Clean verb/noun abstraction-gradient Logit Lens probe."""

from __future__ import annotations

import argparse
import math
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
DEFAULT_STEMS_PATH = DATA_DIR / "abstraction_gradient_clean_stems.csv"
DEFAULT_WORDS_PATH = DATA_DIR / "abstraction_gradient_clean_word_groups.csv"

MIDDLE_LAYERS = [3, 4, 5, 6]
LATE_LAYERS = [8, 9, 10, 11]

GROUP_COLORS = {
    "concrete_practical": "#c2410c",
    "social_emotional": "#ca8a04",
    "reflective_abstract": "#2563eb",
    "existential_identity": "#7c3aed",
}

GROUP_LABELS = {
    "concrete_practical": "1 concrete/practical",
    "social_emotional": "2 social/emotional",
    "reflective_abstract": "3 reflective/abstract",
    "existential_identity": "4 existential/identity",
}


@dataclass(frozen=True)
class WordSpec:
    abstraction_level: int
    word_group: str
    word: str
    lexical_type: str
    syntactic_frame: str
    role: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=48)
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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


def prepare_tokenizer_and_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return tokenizer, model


def clean_word_specs() -> pd.DataFrame:
    verb_specs = {
        1: ("concrete_practical", "practical_action", ["buy", "check", "call", "clean", "order", "fix", "pack", "pay", "cook", "drive"]),
        2: ("social_emotional", "social_emotional_action", ["help", "comfort", "trust", "forgive", "support", "share", "listen", "love", "care", "thank"]),
        3: ("reflective_abstract", "reflective_abstract_action", ["understand", "reflect", "question", "learn", "think", "reconsider", "analyze", "imagine", "explore", "clarify"]),
        4: ("existential_identity", "existential_identity_action", ["become", "belong", "live", "matter", "exist", "grow", "transform", "define", "discover", "choose"]),
    }
    noun_specs = {
        1: ("concrete_practical", "practical_object", ["phone", "room", "order", "bill", "car", "bag", "list", "email", "report", "schedule"]),
        2: ("social_emotional", "social_emotional_domain", ["friend", "family", "partner", "feelings", "relationship", "trust", "conflict", "conversation", "support", "care"]),
        3: ("reflective_abstract", "reflective_abstract_domain", ["thoughts", "question", "memory", "choices", "reason", "lesson", "idea", "belief", "reflection", "understanding"]),
        4: ("existential_identity", "existential_identity_domain", ["life", "identity", "purpose", "future", "values", "meaning", "story", "role", "existence", "mortality"]),
    }
    rows: list[WordSpec] = []
    for level, (group, role, words) in verb_specs.items():
        rows.extend(
            WordSpec(level, group, word, "verb", "verb_after_to", role)
            for word in words
        )
    for level, (group, role, words) in noun_specs.items():
        rows.extend(
            WordSpec(level, group, word, "noun", "noun_after_possessive", role)
            for word in words
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def clean_stems() -> pd.DataFrame:
    names = [
        ("John", "his"),
        ("Mary", "her"),
        ("Alex", "his"),
        ("Sarah", "her"),
        ("David", "his"),
        ("Emma", "her"),
    ]
    rows = []
    verb_templates = [
        ("wants_to", "{name} wants to"),
        ("hopes_to", "{name} hopes to"),
        ("tries_to", "{name} tries to"),
        ("needs_to", "{name} needs to"),
        ("decided_to", "{name} decided to"),
        ("plans_to", "{name} plans to"),
    ]
    noun_templates = [
        ("thinks_about_possessive", "{name} thinks about {poss}"),
        ("reflects_on_possessive", "{name} reflects on {poss}"),
        ("considers_possessive", "{name} considers {poss}"),
        ("questions_possessive", "{name} questions {poss}"),
        ("examines_possessive", "{name} examines {poss}"),
    ]
    for name, _poss in names:
        for template_id, template in verb_templates:
            rows.append(
                {
                    "stem_id": f"verb_{template_id}_{name.lower()}",
                    "name": name,
                    "possessive": "",
                    "stem_template": template_id,
                    "stem": template.format(name=name),
                    "lexical_type": "verb",
                    "syntactic_frame": "verb_after_to",
                }
            )
    for name, poss in names:
        for template_id, template in noun_templates:
            rows.append(
                {
                    "stem_id": f"noun_{template_id}_{name.lower()}",
                    "name": name,
                    "possessive": poss,
                    "stem_template": template_id,
                    "stem": template.format(name=name, poss=poss),
                    "lexical_type": "noun",
                    "syntactic_frame": "noun_after_possessive",
                }
            )
    return pd.DataFrame(rows)


def token_check_words(words: pd.DataFrame, tokenizer) -> pd.DataFrame:
    rows = []
    for _, row in words.iterrows():
        token_text = " " + str(row["word"])
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        rows.append(
            {
                **row.to_dict(),
                "token_text": token_text,
                "token_ids": " ".join(str(token_id) for token_id in token_ids),
                "token_id": int(token_ids[0]) if len(token_ids) == 1 else np.nan,
                "is_single_token": len(token_ids) == 1,
            }
        )
    checked = pd.DataFrame(rows)
    checked["is_usable_primary"] = checked["is_single_token"]
    usable = checked[checked["is_usable_primary"]]
    counts = usable.groupby(["lexical_type", "word_group"]).size().unstack(fill_value=0)
    if (counts < 6).any().any():
        raise ValueError(f"Too few usable single-token candidates in at least one cell:\n{counts}")
    return checked


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
        batches.append(np.stack(per_layer, axis=0))
    return np.concatenate(batches, axis=1).transpose(1, 0, 2)


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_metrics(
    stems: pd.DataFrame,
    words: pd.DataFrame,
    logits: np.ndarray,
    token_ids: list[int],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_to_col = {token_id: col for col, token_id in enumerate(token_ids)}
    group_rows = []
    score_rows = []
    word_rows = []
    final_layer = logits.shape[1] - 1

    group_order = (
        words[["abstraction_level", "word_group"]]
        .drop_duplicates()
        .sort_values("abstraction_level")["word_group"]
        .tolist()
    )
    level_by_group = words.drop_duplicates("word_group").set_index("word_group")["abstraction_level"].to_dict()

    for stem_index, stem in stems.iterrows():
        frame_words = words[words["syntactic_frame"].eq(stem["syntactic_frame"])].reset_index(drop=True)
        frame_cols = [id_to_col[int(token_id)] for token_id in frame_words["token_id"]]
        frame_levels = frame_words["abstraction_level"].to_numpy(dtype=float)
        high_mask = frame_words["abstraction_level"].to_numpy() >= 3
        low_mask = frame_words["abstraction_level"].to_numpy() <= 2

        for layer_index in range(logits.shape[1]):
            values = logits[stem_index, layer_index, frame_cols]
            probs = softmax(values)
            relative_depth = layer_index / final_layer if final_layer else 0.0
            base = {
                "model_name": model_name,
                "stem_id": stem["stem_id"],
                "name": stem["name"],
                "stem_template": stem["stem_template"],
                "stem": stem["stem"],
                "lexical_type": stem["lexical_type"],
                "syntactic_frame": stem["syntactic_frame"],
                "layer_index": layer_index,
                "relative_depth": relative_depth,
                "layer_name": "embedding" if layer_index == 0 else f"block_{layer_index}",
            }
            group_lses = {}
            group_probs = {}
            for group in group_order:
                group_mask = frame_words["word_group"].to_numpy() == group
                group_lses[group] = logsumexp(values[group_mask])
                group_probs[group] = float(probs[group_mask].sum())
                group_rows.append(
                    {
                        **base,
                        "abstraction_level": int(level_by_group[group]),
                        "word_group": group,
                        "word_group_label": GROUP_LABELS[group],
                        "group_logsumexp": group_lses[group],
                        "group_candidate_probability": group_probs[group],
                        "n_group_words": int(group_mask.sum()),
                    }
                )
            expected_level = float(np.sum(probs * frame_levels))
            high_lse = logsumexp(values[high_mask])
            low_lse = logsumexp(values[low_mask])
            high_prob = float(probs[high_mask].sum())
            low_prob = float(probs[low_mask].sum())
            score_rows.append(
                {
                    **base,
                    "expected_abstraction_level": expected_level,
                    "normalized_abstraction_score": (expected_level - 1.0) / 3.0,
                    "high_abstraction_probability": high_prob,
                    "low_abstraction_probability": low_prob,
                    "high_minus_low_probability": high_prob - low_prob,
                    "high_abstraction_logsumexp": high_lse,
                    "low_abstraction_logsumexp": low_lse,
                    "high_minus_low_logit_margin": high_lse - low_lse,
                }
            )
            for word_index, word in frame_words.iterrows():
                word_rows.append(
                    {
                        **base,
                        "abstraction_level": int(word["abstraction_level"]),
                        "word_group": word["word_group"],
                        "word_group_label": GROUP_LABELS[word["word_group"]],
                        "word": word["word"],
                        "role": word["role"],
                        "token_text": word["token_text"],
                        "token_id": int(word["token_id"]),
                        "word_logit": float(values[word_index]),
                        "candidate_set_probability": float(probs[word_index]),
                    }
                )
    return pd.DataFrame(group_rows), pd.DataFrame(score_rows), pd.DataFrame(word_rows)


def band_layers(final_layer: int) -> dict[str, list[int]]:
    middle = [layer for layer in MIDDLE_LAYERS if layer <= final_layer]
    late = [layer for layer in LATE_LAYERS if layer < final_layer]
    if not middle:
        middle = [max(1, int(round(final_layer * 0.4)))]
    if not late:
        late = [max(1, final_layer - 1)]
    return {"middle": middle, "late": late, "final": [final_layer]}


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
        return {"n_stems": n, "mean": float(np.nanmean(arr)) if n else np.nan, "sd": np.nan, "t": np.nan, "df": n - 1, "p_two_sided": np.nan}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n)
    t_value = mean / se if se > 0 else float("nan")
    df = n - 1
    p_value = 2 * (1 - student_t_cdf(abs(t_value), df)) if df > 0 else float("nan")
    return {"n_stems": n, "mean": mean, "sd": sd, "t": t_value, "df": df, "p_two_sided": p_value}


def summarize(group_metrics: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    by_group_layer = (
        group_metrics.groupby(["model_name", "abstraction_level", "word_group", "word_group_label", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_group_probability=("group_candidate_probability", "mean"),
            sd_group_probability=("group_candidate_probability", "std"),
            sem_group_probability=("group_candidate_probability", "sem"),
            mean_group_logsumexp=("group_logsumexp", "mean"),
            n_observations=("group_candidate_probability", "size"),
        )
        .reset_index()
    )
    by_lexical_group_layer = (
        group_metrics.groupby(["model_name", "lexical_type", "abstraction_level", "word_group", "word_group_label", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_group_probability=("group_candidate_probability", "mean"),
            mean_group_logsumexp=("group_logsumexp", "mean"),
            n_observations=("group_candidate_probability", "size"),
        )
        .reset_index()
    )
    by_template_group_layer = (
        group_metrics.groupby(["model_name", "lexical_type", "stem_template", "abstraction_level", "word_group", "word_group_label", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_group_probability=("group_candidate_probability", "mean"),
            mean_group_logsumexp=("group_logsumexp", "mean"),
            n_observations=("group_candidate_probability", "size"),
        )
        .reset_index()
    )
    score_by_layer = (
        scores.groupby(["model_name", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_expected_abstraction_level=("expected_abstraction_level", "mean"),
            sd_expected_abstraction_level=("expected_abstraction_level", "std"),
            sem_expected_abstraction_level=("expected_abstraction_level", "sem"),
            mean_normalized_abstraction_score=("normalized_abstraction_score", "mean"),
            mean_high_abstraction_probability=("high_abstraction_probability", "mean"),
            mean_low_abstraction_probability=("low_abstraction_probability", "mean"),
            mean_high_minus_low_probability=("high_minus_low_probability", "mean"),
            mean_high_minus_low_logit_margin=("high_minus_low_logit_margin", "mean"),
            sd_high_minus_low_logit_margin=("high_minus_low_logit_margin", "std"),
            sem_high_minus_low_logit_margin=("high_minus_low_logit_margin", "sem"),
            n_observations=("expected_abstraction_level", "size"),
        )
        .reset_index()
    )
    score_by_lexical_layer = (
        scores.groupby(["model_name", "lexical_type", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_expected_abstraction_level=("expected_abstraction_level", "mean"),
            mean_high_minus_low_logit_margin=("high_minus_low_logit_margin", "mean"),
            mean_high_abstraction_probability=("high_abstraction_probability", "mean"),
            n_observations=("expected_abstraction_level", "size"),
        )
        .reset_index()
    )
    score_by_template_layer = (
        scores.groupby(["model_name", "lexical_type", "stem_template", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_expected_abstraction_level=("expected_abstraction_level", "mean"),
            mean_high_minus_low_logit_margin=("high_minus_low_logit_margin", "mean"),
            n_observations=("expected_abstraction_level", "size"),
        )
        .reset_index()
    )
    return (
        by_group_layer,
        by_lexical_group_layer,
        by_template_group_layer,
        score_by_layer,
        score_by_lexical_layer,
        score_by_template_layer,
    )


def summarize_bands(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_layer = int(scores["layer_index"].max())
    bands = band_layers(final_layer)
    rows = []
    tests = []
    grouping_specs = [
        ("all", []),
        ("lexical_type", ["lexical_type"]),
    ]
    for scope, extra_cols in grouping_specs:
        for keys, group in scores.groupby(extra_cols, sort=True) if extra_cols else [((), scores)]:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_values = dict(zip(extra_cols, keys))
            pivot_expected = group.pivot_table(index="stem_id", columns="layer_index", values="expected_abstraction_level")
            pivot_margin = group.pivot_table(index="stem_id", columns="layer_index", values="high_minus_low_logit_margin")
            expected_middle = pivot_expected[[layer for layer in bands["middle"] if layer in pivot_expected.columns]].mean(axis=1)
            expected_late = pivot_expected[[layer for layer in bands["late"] if layer in pivot_expected.columns]].mean(axis=1)
            expected_final = pivot_expected[[layer for layer in bands["final"] if layer in pivot_expected.columns]].mean(axis=1)
            margin_middle = pivot_margin[[layer for layer in bands["middle"] if layer in pivot_margin.columns]].mean(axis=1)
            margin_late = pivot_margin[[layer for layer in bands["late"] if layer in pivot_margin.columns]].mean(axis=1)
            margin_final = pivot_margin[[layer for layer in bands["final"] if layer in pivot_margin.columns]].mean(axis=1)
            row = {
                "scope": scope,
                **key_values,
                "n_stems": int(len(pivot_expected)),
                "middle_layers": " ".join(str(layer) for layer in bands["middle"]),
                "late_layers": " ".join(str(layer) for layer in bands["late"]),
                "final_layer": final_layer,
                "expected_middle_mean": float(expected_middle.mean()),
                "expected_late_mean": float(expected_late.mean()),
                "expected_final_mean": float(expected_final.mean()),
                "expected_late_minus_middle": float((expected_late - expected_middle).mean()),
                "expected_late_minus_final": float((expected_late - expected_final).mean()),
                "margin_middle_mean": float(margin_middle.mean()),
                "margin_late_mean": float(margin_late.mean()),
                "margin_final_mean": float(margin_final.mean()),
                "margin_late_minus_middle": float((margin_late - margin_middle).mean()),
                "margin_late_minus_final": float((margin_late - margin_final).mean()),
            }
            rows.append(row)
            for metric_name, values in [
                ("expected_late_minus_middle", expected_late - expected_middle),
                ("margin_late_minus_middle", margin_late - margin_middle),
            ]:
                test = paired_test(values)
                tests.append(
                    {
                        "scope": scope,
                        **key_values,
                        "metric": metric_name,
                        **test,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(tests)


def plot_group_probability(by_group_layer: pd.DataFrame, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=180)
    for group, sub in by_group_layer.groupby("word_group", sort=False):
        sub = sub.sort_values("layer_index")
        ax.plot(
            sub["layer_index"],
            sub["mean_group_probability"],
            marker="o",
            linewidth=2.4,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group],
        )
    ax.set_title("Clean abstraction gradient: probability mass by layer")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Probability within grammar-matched candidate set")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.24)
    ax.legend(title="Candidate level")
    fig.tight_layout()
    fig.savefig(plot_dir / "abstraction_gradient_clean_probability_by_layer.png", bbox_inches="tight")
    plt.close(fig)


def plot_stacked_probability(by_group_layer: pd.DataFrame, plot_dir: Path) -> None:
    table = by_group_layer.pivot(index="layer_index", columns="word_group", values="mean_group_probability")
    order = ["concrete_practical", "social_emotional", "reflective_abstract", "existential_identity"]
    fig, ax = plt.subplots(figsize=(10.0, 5.4), dpi=180)
    ax.stackplot(
        table.index,
        [table[group] for group in order],
        labels=[GROUP_LABELS[group] for group in order],
        colors=[GROUP_COLORS[group] for group in order],
        alpha=0.82,
    )
    ax.set_title("Clean abstraction gradient: stacked candidate probability")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Probability mass")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "abstraction_gradient_clean_stacked_probability_by_layer.png", bbox_inches="tight")
    plt.close(fig)


def plot_expected_and_margin(score_by_layer: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    x = score_by_layer["layer_index"].to_numpy(dtype=float)
    y = score_by_layer["mean_expected_abstraction_level"].to_numpy(dtype=float)
    sem = score_by_layer["sem_expected_abstraction_level"].fillna(0).to_numpy(dtype=float)
    axes[0].plot(x, y, marker="o", linewidth=2.6, color="#111827")
    axes[0].fill_between(x, y - sem, y + sem, color="#111827", alpha=0.12)
    axes[0].axvspan(8, 11, color="#2563eb", alpha=0.07, label="late band")
    axes[0].set_title("Expected abstraction score")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Expected level (1 concrete to 4 existential)")
    axes[0].set_ylim(1, 4)
    axes[0].grid(True, alpha=0.24)
    axes[0].legend()

    margin = score_by_layer["mean_high_minus_low_logit_margin"].to_numpy(dtype=float)
    margin_sem = score_by_layer["sem_high_minus_low_logit_margin"].fillna(0).to_numpy(dtype=float)
    axes[1].plot(x, margin, marker="o", linewidth=2.6, color="#111827")
    axes[1].fill_between(x, margin - margin_sem, margin + margin_sem, color="#111827", alpha=0.12)
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].axvspan(8, 11, color="#2563eb", alpha=0.07, label="late band")
    axes[1].set_title("High-abstraction minus low-abstraction margin")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel("Logsumexp margin: levels 3-4 minus 1-2")
    axes[1].grid(True, alpha=0.24)
    axes[1].legend()
    fig.suptitle("Clean abstraction-gradient layer scores", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "abstraction_gradient_clean_expected_and_margin_by_layer.png", bbox_inches="tight")
    plt.close(fig)


def heatmap(table: pd.DataFrame, title: str, ylabel: str, output_path: Path, cmap: str = "viridis", center_zero: bool = False) -> None:
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.36 * len(table) + 1.8)), dpi=180)
    if center_zero:
        limit = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 0.01)
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    ax.set_title(title)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(
    by_group_layer: pd.DataFrame,
    score_by_lexical_layer: pd.DataFrame,
    score_by_template_layer: pd.DataFrame,
    plot_dir: Path,
) -> None:
    group_table = by_group_layer.pivot(index="word_group_label", columns="layer_index", values="mean_group_probability")
    order = [GROUP_LABELS[group] for group in ["concrete_practical", "social_emotional", "reflective_abstract", "existential_identity"]]
    heatmap(group_table.loc[order], "Probability mass by abstraction level", "Candidate level", plot_dir / "abstraction_gradient_clean_group_layer_heatmap.png")

    lexical_table = score_by_lexical_layer.pivot(index="lexical_type", columns="layer_index", values="mean_expected_abstraction_level")
    heatmap(lexical_table, "Expected abstraction score by lexical type", "Lexical type", plot_dir / "abstraction_gradient_clean_lexical_type_heatmap.png")

    template_table = score_by_template_layer.pivot_table(index=["lexical_type", "stem_template"], columns="layer_index", values="mean_expected_abstraction_level")
    template_table.index = [f"{lex}: {template}" for lex, template in template_table.index]
    order = (
        score_by_template_layer[score_by_template_layer["layer_index"].isin(LATE_LAYERS)]
        .groupby(["lexical_type", "stem_template"])["mean_expected_abstraction_level"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    ordered_labels = [f"{lex}: {template}" for lex, template in order]
    heatmap(template_table.loc[ordered_labels], "Expected abstraction score by stem template", "Stem template", plot_dir / "abstraction_gradient_clean_template_heatmap.png")

    margin_table = score_by_template_layer.pivot_table(index=["lexical_type", "stem_template"], columns="layer_index", values="mean_high_minus_low_logit_margin")
    margin_table.index = [f"{lex}: {template}" for lex, template in margin_table.index]
    heatmap(margin_table.loc[ordered_labels], "High-minus-low margin by stem template", "Stem template", plot_dir / "abstraction_gradient_clean_template_margin_heatmap.png", center_zero=True)


def plot_word_heatmap(word_metrics: pd.DataFrame, plot_dir: Path) -> None:
    collapsed = (
        word_metrics.groupby(["lexical_type", "word_group", "abstraction_level", "word", "layer_index"], sort=False)
        .agg(mean_candidate_probability=("candidate_set_probability", "mean"))
        .reset_index()
    )
    collapsed["word_label"] = collapsed["lexical_type"] + ": " + collapsed["word"] + " (" + collapsed["word_group"] + ")"
    order = (
        collapsed[collapsed["layer_index"].isin(LATE_LAYERS)]
        .groupby("word_label")["mean_candidate_probability"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    table = collapsed.pivot_table(index="word_label", columns="layer_index", values="mean_candidate_probability", aggfunc="mean").loc[order]
    heatmap(table, "Individual candidate probabilities", "Candidate word", plot_dir / "abstraction_gradient_clean_word_heatmap.png")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    tokenizer, model = prepare_tokenizer_and_model(args.model_name, device)
    stems = clean_stems()
    words_raw = clean_word_specs()
    token_check = token_check_words(words_raw, tokenizer)
    words = token_check[token_check["is_usable_primary"]].copy().reset_index(drop=True)

    stems.to_csv(DEFAULT_STEMS_PATH, index=False)
    words_raw.to_csv(DEFAULT_WORDS_PATH, index=False)
    token_check.to_csv(output_dir / "abstraction_gradient_clean_token_check.csv", index=False)

    token_ids = sorted(int(token_id) for token_id in words["token_id"].unique())
    print(f"Scoring {len(stems)} stems and {len(words)} usable single-token candidates with {args.model_name} on {device}...")
    logits = layer_logits_for_tokens(
        model=model,
        tokenizer=tokenizer,
        stems=stems["stem"].tolist(),
        token_ids=token_ids,
        device=device,
        batch_size=args.batch_size,
    )
    group_metrics, scores, word_metrics = build_metrics(stems, words, logits, token_ids, args.model_name)
    (
        by_group_layer,
        by_lexical_group_layer,
        by_template_group_layer,
        score_by_layer,
        score_by_lexical_layer,
        score_by_template_layer,
    ) = summarize(group_metrics, scores)
    band_summary, test_summary = summarize_bands(scores)

    group_metrics.to_csv(output_dir / "abstraction_gradient_clean_group_metrics.csv", index=False)
    scores.to_csv(output_dir / "abstraction_gradient_clean_stem_layer_scores.csv", index=False)
    word_metrics.to_csv(output_dir / "abstraction_gradient_clean_word_metrics.csv", index=False)
    by_group_layer.to_csv(output_dir / "abstraction_gradient_clean_summary_by_group_layer.csv", index=False)
    by_lexical_group_layer.to_csv(output_dir / "abstraction_gradient_clean_summary_by_lexical_group_layer.csv", index=False)
    by_template_group_layer.to_csv(output_dir / "abstraction_gradient_clean_summary_by_template_group_layer.csv", index=False)
    score_by_layer.to_csv(output_dir / "abstraction_gradient_clean_score_by_layer.csv", index=False)
    score_by_lexical_layer.to_csv(output_dir / "abstraction_gradient_clean_score_by_lexical_type_layer.csv", index=False)
    score_by_template_layer.to_csv(output_dir / "abstraction_gradient_clean_score_by_template_layer.csv", index=False)
    band_summary.to_csv(output_dir / "abstraction_gradient_clean_band_summary.csv", index=False)
    test_summary.to_csv(output_dir / "abstraction_gradient_clean_test_summary.csv", index=False)

    plot_group_probability(by_group_layer, plot_dir)
    plot_stacked_probability(by_group_layer, plot_dir)
    plot_expected_and_margin(score_by_layer, plot_dir)
    plot_heatmaps(by_group_layer, score_by_lexical_layer, score_by_template_layer, plot_dir)
    plot_word_heatmap(word_metrics, plot_dir)

    print("\nDone.")
    print(f"Wrote data to {DEFAULT_STEMS_PATH} and {DEFAULT_WORDS_PATH}")
    print(f"Wrote CSVs to {output_dir}")
    print(f"Wrote plots to {plot_dir}")
    print("\nBand summary:")
    print(band_summary.to_string(index=False))
    print("\nTests:")
    print(test_summary.to_string(index=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

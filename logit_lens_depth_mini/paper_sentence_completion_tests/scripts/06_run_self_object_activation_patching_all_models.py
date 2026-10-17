#!/usr/bin/env python3
"""Cross-model activation patching for the self-directed vs object-directed probe."""

from __future__ import annotations

import argparse
import gc
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
CORE_INNER_WORDS = {"life", "values", "identity", "future", "feelings", "purpose"}
CORE_OBJECT_WORDS = {"phone", "schedule", "order", "room", "email", "list"}
MIDDLE_MIN_RELATIVE = 0.25
MIDDLE_MAX_RELATIVE = 0.50
LATE_MIN_RELATIVE = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
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


def blocks_for_model(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError(f"Unsupported model architecture: {type(model)}")


def attention_module(block):
    if hasattr(block, "attn"):
        return block.attn
    if hasattr(block, "attention"):
        return block.attention
    return None


def mlp_module(block):
    return block.mlp if hasattr(block, "mlp") else None


def output_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    final_position = int(attention_mask.sum().item()) - 1
    return outputs.logits[0, final_position, :]


def load_pairs(max_pairs: int, seed: int) -> pd.DataFrame:
    stems = pd.read_csv(DATA_DIR / "self_object_stems.csv")
    pair_ids = np.array(sorted(stems["pair_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(pair_ids)
    if max_pairs > 0:
        stems = stems[stems["pair_id"].isin(pair_ids[:max_pairs])].copy()
    return stems.sort_values(["pair_id", "stem_family"]).reset_index(drop=True)


def load_words(tokenizer, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    words = pd.read_csv(DATA_DIR / "self_object_word_groups.csv")
    rows = []
    token_rows = []
    for _, row in words.iterrows():
        token_text = " " + str(row["word"])
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        is_single = len(token_ids) == 1
        token_rows.append({"model_name": model_name, **row.to_dict(), "token_ids": " ".join(str(x) for x in token_ids), "is_single_token": is_single})
        if not is_single:
            continue
        word = str(row["word"]).lower()
        keep = (row["word_group"] == "inner" and word in CORE_INNER_WORDS) or (
            row["word_group"] == "object" and word in CORE_OBJECT_WORDS
        )
        if keep:
            rows.append({**row.to_dict(), "token_text": token_text, "token_id": int(token_ids[0])})
    selected = pd.DataFrame(rows)
    if selected.groupby("word_group").size().min() < 1:
        raise ValueError("Core candidate set became empty after tokenization.")
    return selected.reset_index(drop=True), pd.DataFrame(token_rows)


def candidate_metrics(logits: torch.Tensor, words: pd.DataFrame) -> dict[str, float]:
    inner_ids = [int(x) for x in words.loc[words["word_group"].eq("inner"), "token_id"]]
    object_ids = [int(x) for x in words.loc[words["word_group"].eq("object"), "token_id"]]
    candidate_ids = inner_ids + object_ids
    candidate_logits = logits[candidate_ids]
    candidate_probs = torch.softmax(candidate_logits, dim=-1)
    n_inner = len(inner_ids)
    inner_lse = torch.logsumexp(logits[inner_ids], dim=0)
    object_lse = torch.logsumexp(logits[object_ids], dim=0)
    return {
        "inner_minus_object_margin": float((inner_lse - object_lse).detach().cpu()),
        "inner_probability": float(candidate_probs[:n_inner].sum().detach().cpu()),
        "object_probability": float(candidate_probs[n_inner:].sum().detach().cpu()),
    }


def encode(tokenizer, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def save_component_output(cache: dict, component: str, layer_index: int):
    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        cache[component][layer_index] = value.detach().clone()
        return output

    return hook


def cache_source(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, list[torch.Tensor]]:
    blocks = blocks_for_model(model)
    cache: dict[str, list[torch.Tensor]] = {
        "attn_out": [None] * len(blocks),  # type: ignore[list-item]
        "mlp_out": [None] * len(blocks),  # type: ignore[list-item]
    }
    handles = []
    for layer_index, block in enumerate(blocks):
        attn = attention_module(block)
        mlp = mlp_module(block)
        if attn is not None:
            handles.append(attn.register_forward_hook(save_component_output(cache, "attn_out", layer_index)))
        if mlp is not None:
            handles.append(mlp.register_forward_hook(save_component_output(cache, "mlp_out", layer_index)))
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    for handle in handles:
        handle.remove()
    cache["resid_pre"] = [hidden.detach().clone() for hidden in outputs.hidden_states[:-1]]
    cache["resid_post"] = [hidden.detach().clone() for hidden in outputs.hidden_states[1:]]
    return cache


def patch_final_token(target_tensor: torch.Tensor, source_tensor: torch.Tensor, target_pos: int, source_pos: int) -> torch.Tensor:
    patched = target_tensor.clone()
    patched[:, target_pos, :] = source_tensor[:, source_pos, :]
    return patched


def patched_logits(
    model,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    source_cache: dict[str, list[torch.Tensor]],
    component: str,
    layer_index: int,
    target_pos: int,
    source_pos: int,
) -> torch.Tensor:
    blocks = blocks_for_model(model)
    block = blocks[layer_index]
    source_tensor = source_cache[component][layer_index]
    if source_tensor is None:
        raise ValueError(f"Missing source tensor for {component} layer {layer_index}")
    handles = []

    if component == "resid_pre":
        def pre_hook(_module, inputs):
            patched = patch_final_token(inputs[0], source_tensor, target_pos, source_pos)
            return (patched,) + inputs[1:]

        handles.append(block.register_forward_pre_hook(pre_hook))
    elif component == "resid_post":
        def post_hook(_module, _inputs, output):
            first = output[0] if isinstance(output, tuple) else output
            patched = patch_final_token(first, source_tensor, target_pos, source_pos)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        handles.append(block.register_forward_hook(post_hook))
    elif component == "attn_out":
        module = attention_module(block)

        def attn_hook(_module, _inputs, output):
            first = output[0] if isinstance(output, tuple) else output
            patched = patch_final_token(first, source_tensor, target_pos, source_pos)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        handles.append(module.register_forward_hook(attn_hook))
    elif component == "mlp_out":
        module = mlp_module(block)

        def mlp_hook(_module, _inputs, output):
            return patch_final_token(output, source_tensor, target_pos, source_pos)

        handles.append(module.register_forward_hook(mlp_hook))
    else:
        raise ValueError(f"Unknown component: {component}")

    with torch.inference_mode():
        outputs = model(input_ids=target_ids, attention_mask=target_mask, use_cache=False)
    for handle in handles:
        handle.remove()
    final_position = int(target_mask.sum().item()) - 1
    return outputs.logits[0, final_position, :]


def run_model(model_name: str, model, tokenizer, stems: pd.DataFrame, words: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    blocks = blocks_for_model(model)
    rows = []
    for pair_id, pair in stems.groupby("pair_id", sort=False):
        self_row = pair[pair["stem_family"].eq("self")].iloc[0]
        object_row = pair[pair["stem_family"].eq("object")].iloc[0]
        source_ids, source_mask = encode(tokenizer, self_row["stem"], device)
        target_ids, target_mask = encode(tokenizer, object_row["stem"], device)
        source_pos = int(source_mask.sum().item()) - 1
        target_pos = int(target_mask.sum().item()) - 1
        source_clean = candidate_metrics(output_logits(model, source_ids, source_mask), words)
        target_logits = output_logits(model, target_ids, target_mask)
        target_clean = candidate_metrics(target_logits, words)
        source_cache = cache_source(model, source_ids, source_mask)
        clean_gap = source_clean["inner_minus_object_margin"] - target_clean["inner_minus_object_margin"]
        for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
            if component in {"attn_out", "mlp_out"} and source_cache[component][0] is None:
                continue
            for layer_index in range(len(blocks)):
                patched = patched_logits(
                    model,
                    target_ids,
                    target_mask,
                    source_cache,
                    component,
                    layer_index,
                    target_pos,
                    source_pos,
                )
                patched_metrics = candidate_metrics(patched, words)
                delta_margin = patched_metrics["inner_minus_object_margin"] - target_clean["inner_minus_object_margin"]
                rows.append(
                    {
                        "model_name": model_name,
                        "model_label": model_label(model_name),
                        "n_transformer_layers": len(blocks),
                        "pair_id": pair_id,
                        "self_stem": self_row["stem"],
                        "object_stem": object_row["stem"],
                        "component": component,
                        "layer_index": layer_index + 1,
                        "relative_depth": (layer_index + 1) / len(blocks),
                        "self_clean_margin": source_clean["inner_minus_object_margin"],
                        "object_clean_margin": target_clean["inner_minus_object_margin"],
                        "clean_gap_self_minus_object": clean_gap,
                        "patched_margin": patched_metrics["inner_minus_object_margin"],
                        "delta_inner_minus_object_margin": delta_margin,
                        "patched_inner_probability": patched_metrics["inner_probability"],
                        "object_clean_inner_probability": target_clean["inner_probability"],
                        "delta_inner_probability": patched_metrics["inner_probability"] - target_clean["inner_probability"],
                        "normalized_margin_effect": delta_margin / clean_gap if abs(clean_gap) > 1e-8 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_effects(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_layer = (
        effects.groupby(["model_name", "model_label", "component", "layer_index", "relative_depth"], sort=True)
        .agg(
            mean_delta_margin=("delta_inner_minus_object_margin", "mean"),
            sem_delta_margin=("delta_inner_minus_object_margin", "sem"),
            mean_delta_inner_probability=("delta_inner_probability", "mean"),
            mean_normalized_margin_effect=("normalized_margin_effect", "mean"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
    )
    rows = []
    for keys, group in effects.groupby(["model_name", "model_label", "component"], sort=False):
        model_name, label, component = keys
        max_layer = int(group["n_transformer_layers"].iloc[0])
        late = group[group["relative_depth"].gt(LATE_MIN_RELATIVE) & group["relative_depth"].lt(1.0)]
        middle = group[
            group["relative_depth"].ge(MIDDLE_MIN_RELATIVE)
            & group["relative_depth"].le(MIDDLE_MAX_RELATIVE)
        ]
        if late.empty:
            late = group[group["layer_index"].eq(max_layer - 1)]
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "component": component,
                "n_pairs": int(group["pair_id"].nunique()),
                "middle_mean_delta_margin": float(middle["delta_inner_minus_object_margin"].mean()),
                "late_mean_delta_margin": float(late["delta_inner_minus_object_margin"].mean()),
                "late_minus_middle_delta_margin": float(late["delta_inner_minus_object_margin"].mean() - middle["delta_inner_minus_object_margin"].mean()),
                "late_mean_delta_inner_probability": float(late["delta_inner_probability"].mean()),
                "late_mean_normalized_margin_effect": float(late["normalized_margin_effect"].mean()),
                "positive_late_causal_effect": bool(late["delta_inner_minus_object_margin"].mean() > 0),
            }
        )
    return by_layer, pd.DataFrame(rows)


def plot_patch_summary(by_layer: pd.DataFrame, plot_dir: Path) -> None:
    for component, selected in by_layer.groupby("component", sort=False):
        fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=180)
        for label, group in selected.groupby("model_label", sort=False):
            group = group.sort_values("relative_depth")
            ax.plot(group["relative_depth"], group["mean_delta_margin"], marker="o", linewidth=2.0, label=label)
        ax.axhline(0, color="#444444", linewidth=1)
        ax.axvspan(LATE_MIN_RELATIVE, 1.0, color="#111827", alpha=0.06)
        ax.set_title(f"Self-to-object patching effect: {component}")
        ax.set_xlabel("Relative depth")
        ax.set_ylabel("Delta inner-minus-object margin")
        ax.grid(True, alpha=0.24)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"causal_patch_{component}_effect_by_relative_depth.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    stems = load_pairs(args.max_pairs, args.seed)
    all_effects = []
    token_checks = []
    errors = []
    for model_name in args.models:
        tokenizer = None
        model = None
        print(f"\nLoading {model_name} on {device}...")
        try:
            tokenizer, model = prepare_tokenizer_and_model(model_name, device)
            words, token_check = load_words(tokenizer, model_name)
            token_check["model_label"] = model_label(model_name)
            token_checks.append(token_check)
            print(f"Patching {stems['pair_id'].nunique()} self/object pairs for {model_name}...")
            effects = run_model(model_name, model, tokenizer, stems, words, device)
            all_effects.append(effects)
        except Exception as exc:
            errors.append({"model_name": model_name, "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")
        finally:
            cleanup_model(model, tokenizer)
    if token_checks:
        pd.concat(token_checks, ignore_index=True, sort=False).to_csv(output_dir / "causal_patch_token_check.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "causal_patch_errors.csv", index=False)
    if not all_effects:
        raise RuntimeError("No activation-patching runs completed.")
    effects = pd.concat(all_effects, ignore_index=True)
    by_layer, summary = summarize_effects(effects)
    effects.to_csv(output_dir / "causal_patch_component_effects.csv", index=False)
    by_layer.to_csv(output_dir / "causal_patch_summary_by_layer.csv", index=False)
    summary.to_csv(output_dir / "causal_patch_late_component_summary.csv", index=False)
    plot_patch_summary(by_layer, plot_dir)
    print("\nActivation patching summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

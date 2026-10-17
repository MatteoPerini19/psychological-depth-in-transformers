#!/usr/bin/env python3
"""Expanded activation-patching controls for the self/object probe."""

from __future__ import annotations

import gc
import importlib.util
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import CONFIG, REPORTS_DIR, RESULTS_DIR, bootstrap_ci, ensure_dirs, layer_bands, model_label, save_table


PATCH_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "06_run_self_object_activation_patching_all_models.py"
spec = importlib.util.spec_from_file_location("patching_base", PATCH_SCRIPT)
patching_base = importlib.util.module_from_spec(spec)
sys.modules["patching_base"] = patching_base
assert spec.loader is not None
spec.loader.exec_module(patching_base)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def candidate_metrics_detail(logits: torch.Tensor, words: pd.DataFrame) -> dict[str, float]:
    inner_ids = [int(x) for x in words.loc[words["word_group"].eq("inner"), "token_id"]]
    object_ids = [int(x) for x in words.loc[words["word_group"].eq("object"), "token_id"]]
    candidate_ids = inner_ids + object_ids
    candidate_logits = logits[candidate_ids]
    candidate_probs = torch.softmax(candidate_logits, dim=-1)
    n_inner = len(inner_ids)
    inner_lse = torch.logsumexp(logits[inner_ids], dim=0)
    object_lse = torch.logsumexp(logits[object_ids], dim=0)
    return {
        "inner_logsumexp": float(inner_lse.detach().cpu()),
        "object_logsumexp": float(object_lse.detach().cpu()),
        "inner_minus_object_margin": float((inner_lse - object_lse).detach().cpu()),
        "inner_probability": float(candidate_probs[:n_inner].sum().detach().cpu()),
        "object_probability": float(candidate_probs[n_inner:].sum().detach().cpu()),
    }


def patch_vector(tensor: torch.Tensor, vector: torch.Tensor, target_pos: int) -> torch.Tensor:
    patched = tensor.clone()
    patched[:, target_pos, :] = vector.reshape(1, -1).to(patched.device, dtype=patched.dtype)
    return patched


def patched_logits_with_vector(model, target_ids, target_mask, vector, component: str, layer_index: int, target_pos: int):
    blocks = patching_base.blocks_for_model(model)
    block = blocks[layer_index]
    handles = []
    if component == "resid_pre":
        def pre_hook(_module, inputs):
            return (patch_vector(inputs[0], vector, target_pos),) + inputs[1:]
        handles.append(block.register_forward_pre_hook(pre_hook))
    elif component == "resid_post":
        def post_hook(_module, _inputs, output):
            first = output[0] if isinstance(output, tuple) else output
            patched = patch_vector(first, vector, target_pos)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched
        handles.append(block.register_forward_hook(post_hook))
    elif component == "attn_out":
        module = patching_base.attention_module(block)
        def attn_hook(_module, _inputs, output):
            first = output[0] if isinstance(output, tuple) else output
            patched = patch_vector(first, vector, target_pos)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched
        handles.append(module.register_forward_hook(attn_hook))
    elif component == "mlp_out":
        module = patching_base.mlp_module(block)
        def mlp_hook(_module, _inputs, output):
            return patch_vector(output, vector, target_pos)
        handles.append(module.register_forward_hook(mlp_hook))
    else:
        raise ValueError(component)
    with torch.inference_mode():
        outputs = model(input_ids=target_ids, attention_mask=target_mask, use_cache=False)
    for handle in handles:
        handle.remove()
    final_position = int(target_mask.sum().item()) - 1
    return outputs.logits[0, final_position, :]


def final_vector(cache: dict, component: str, layer_index: int, source_pos: int) -> torch.Tensor:
    tensor = cache[component][layer_index]
    return tensor[:, source_pos, :].detach().clone().squeeze(0)


def load_all_pairs() -> pd.DataFrame:
    stems = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "self_object_stems.csv")
    return stems.sort_values(["pair_id", "stem_family"]).reset_index(drop=True)


def encode_prompt(tokenizer, text: str, device: torch.device):
    return patching_base.encode(tokenizer, text, device)


def make_row(
    *,
    model_name: str,
    pair_id,
    control_type: str,
    component: str,
    layer_index_zero: int,
    n_layers: int,
    source_row: pd.Series,
    target_row: pd.Series,
    source_metrics: dict,
    target_clean: dict,
    patched: dict,
    source_pos: int,
    target_pos: int,
    patched_position: str,
) -> dict:
    clean_gap = source_metrics["inner_minus_object_margin"] - target_clean["inner_minus_object_margin"]
    delta_margin = patched["inner_minus_object_margin"] - target_clean["inner_minus_object_margin"]
    denom_ok = abs(clean_gap) > 1e-6
    return {
        "model_name": model_name,
        "model_label": model_label(model_name),
        "pair_id": pair_id,
        "control_type": control_type,
        "component": component,
        "layer_index": layer_index_zero + 1,
        "relative_depth": (layer_index_zero + 1) / n_layers,
        "source_stem_family": source_row["stem_family"],
        "target_stem_family": target_row["stem_family"],
        "source_prompt": source_row["stem"],
        "target_prompt": target_row["stem"],
        "source_pair_id": source_row["pair_id"],
        "target_pair_id": target_row["pair_id"],
        "patched_position": patched_position,
        "source_position": source_pos,
        "target_position": target_pos,
        "source_clean_margin": source_metrics["inner_minus_object_margin"],
        "target_clean_margin": target_clean["inner_minus_object_margin"],
        "patched_margin": patched["inner_minus_object_margin"],
        "raw_delta_margin": delta_margin,
        "clean_corrupted_gap": clean_gap,
        "normalized_recovery": delta_margin / clean_gap if denom_ok else np.nan,
        "normalized_recovery_denominator_ok": denom_ok,
        "target_logsumexp_change": patched["inner_logsumexp"] - target_clean["inner_logsumexp"],
        "contrast_logsumexp_change": patched["object_logsumexp"] - target_clean["object_logsumexp"],
        "patched_inner_probability": patched["inner_probability"],
        "target_clean_inner_probability": target_clean["inner_probability"],
        "delta_inner_probability": patched["inner_probability"] - target_clean["inner_probability"],
    }


def run_model(model_name: str, device: torch.device, rng: np.random.Generator) -> pd.DataFrame:
    tokenizer, model = patching_base.prepare_tokenizer_and_model(model_name, device)
    words, _token_check = patching_base.load_words(tokenizer, model_name)
    stems = load_all_pairs()
    blocks = patching_base.blocks_for_model(model)
    n_layers = len(blocks)
    rows = []

    prompt_cache = {}
    for _, row in stems.iterrows():
        ids, mask = encode_prompt(tokenizer, row["stem"], device)
        prompt_cache[row["stem_id"]] = {
            "row": row,
            "ids": ids,
            "mask": mask,
            "pos": int(mask.sum().item()) - 1,
            "clean_logits": patching_base.output_logits(model, ids, mask),
            "cache": patching_base.cache_source(model, ids, mask),
        }
        prompt_cache[row["stem_id"]]["metrics"] = candidate_metrics_detail(prompt_cache[row["stem_id"]]["clean_logits"], words)

    self_rows = stems[stems["stem_family"].eq("self")].set_index("pair_id", drop=False)
    object_rows = stems[stems["stem_family"].eq("object")].set_index("pair_id", drop=False)
    pair_ids = list(self_rows.index)
    shuffled_sources = pair_ids.copy()
    rng.shuffle(shuffled_sources)
    random_assignments = []
    for _ in range(3):
        assigned = pair_ids.copy()
        rng.shuffle(assigned)
        random_assignments.append(dict(zip(pair_ids, assigned)))
    shuffle_assignment = dict(zip(pair_ids, shuffled_sources))

    mean_vectors = {"resid_post": [None] * n_layers}
    for layer in range(n_layers):
        vectors = []
        for pair_id in pair_ids:
            self_row = self_rows.loc[pair_id]
            src = prompt_cache[self_row["stem_id"]]
            vectors.append(final_vector(src["cache"], "resid_post", layer, src["pos"]))
        mean_vectors["resid_post"][layer] = torch.stack(vectors, dim=0).mean(dim=0)

    def add_patch(control_type: str, component: str, layer: int, source_row: pd.Series, target_row: pd.Series, source_pos_offset: int = 0, target_pos_offset: int = 0, vector=None):
        source = prompt_cache[source_row["stem_id"]]
        target = prompt_cache[target_row["stem_id"]]
        source_pos = source["pos"] + source_pos_offset
        target_pos = target["pos"] + target_pos_offset
        if source_pos < 0 or target_pos < 0:
            return
        if vector is None:
            logits = patching_base.patched_logits(
                model,
                target["ids"],
                target["mask"],
                source["cache"],
                component,
                layer,
                target_pos,
                source_pos,
            )
            source_metrics = source["metrics"]
            patched_position = "final_token" if source_pos_offset == 0 and target_pos_offset == 0 else "one_token_before_final"
        else:
            logits = patched_logits_with_vector(model, target["ids"], target["mask"], vector, component, layer, target_pos)
            source_metrics = source["metrics"]
            patched_position = "mean_activation_final_token"
        patched = candidate_metrics_detail(logits, words)
        rows.append(
            make_row(
                model_name=model_name,
                pair_id=target_row["pair_id"],
                control_type=control_type,
                component=component,
                layer_index_zero=layer,
                n_layers=n_layers,
                source_row=source_row,
                target_row=target_row,
                source_metrics=source_metrics,
                target_clean=target["metrics"],
                patched=patched,
                source_pos=source_pos,
                target_pos=target_pos,
                patched_position=patched_position,
            )
        )

    for pair_id in pair_ids:
        self_row = self_rows.loc[pair_id]
        object_row = object_rows.loc[pair_id]
        for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
            for layer in range(n_layers):
                add_patch("matched_forward_self_to_object", component, layer, self_row, object_row)
                add_patch("reverse_object_to_self", component, layer, object_row, self_row)
        for layer in range(n_layers):
            for assignment_index, assignment in enumerate(random_assignments):
                add_patch(f"random_pair_forward_{assignment_index}", "resid_post", layer, self_rows.loc[assignment[pair_id]], object_row)
            add_patch("source_label_shuffle_forward", "resid_post", layer, self_rows.loc[shuffle_assignment[pair_id]], object_row)
            next_pair = pair_ids[(pair_ids.index(pair_id) + 1) % len(pair_ids)]
            add_patch("same_condition_self_to_self", "resid_post", layer, self_rows.loc[next_pair], self_row)
            add_patch("same_condition_object_to_object", "resid_post", layer, object_rows.loc[next_pair], object_row)
            add_patch("mean_self_activation_to_object", "resid_post", layer, self_row, object_row, vector=mean_vectors["resid_post"][layer])
            add_patch("position_control_one_before", "resid_post", layer, self_row, object_row, source_pos_offset=-1, target_pos_offset=-1)

    del model
    del tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return pd.DataFrame(rows)


def summarize(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in effects.groupby(["model_name", "model_label", "control_type", "component"], sort=False):
        model_name, label, control_type, component = keys
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        middle = group[group["layer_index"].isin(bands["middle"])]["raw_delta_margin"]
        late = group[group["layer_index"].isin(bands["late"])]["raw_delta_margin"]
        final = group[group["layer_index"].eq(n_layers)]["raw_delta_margin"]
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "control_type": control_type,
                "component": component,
                "n_pairs": int(group["target_pair_id"].nunique()),
                "middle_mean_delta": float(middle.mean()),
                "late_mean_delta": float(late.mean()),
                "final_mean_delta": float(final.mean()),
                "late_minus_middle_delta": float(late.mean() - middle.mean()),
                "late_mean_normalized_recovery": float(group[group["layer_index"].isin(bands["late"]) & group["normalized_recovery_denominator_ok"]]["normalized_recovery"].mean()),
                "late_target_logsumexp_change": float(group[group["layer_index"].isin(bands["late"])]["target_logsumexp_change"].mean()),
                "late_contrast_logsumexp_change": float(group[group["layer_index"].isin(bands["late"])]["contrast_logsumexp_change"].mean()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(effects: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n_boot = int(CONFIG["patching_bootstrap_resamples"])
    for keys, group in effects.groupby(["model_name", "model_label", "control_type", "component"], sort=False):
        model_name, label, control_type, component = keys
        n_layers = int(group["layer_index"].max())
        bands = layer_bands(n_layers)
        late = group[group["layer_index"].isin(bands["late"])]
        unit = late.groupby("target_pair_id")["raw_delta_margin"].mean().dropna()
        if len(unit) < 2:
            continue
        values = np.empty(n_boot, dtype=float)
        ids = unit.index.to_numpy()
        for i in range(n_boot):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            values[i] = float(unit.loc[sampled].mean())
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "model_name": model_name,
                "model_label": label,
                "control_type": control_type,
                "component": component,
                "late_mean_delta": float(unit.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_positive_probability": float(np.mean(values > 0)),
                "n_bootstrap": n_boot,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=CONFIG["models"])
    args = parser.parse_args()
    ensure_dirs()
    device = choose_device()
    rng = np.random.default_rng(int(CONFIG["patching_seed"]))
    frames = []
    errors = []
    for model_name in args.models:
        try:
            print(f"Running patching controls for {model_name} on {device}...")
            frames.append(run_model(model_name, device, rng))
        except Exception as exc:
            errors.append({"model_name": model_name, "error": repr(exc)})
            print(f"ERROR for {model_name}: {exc}")
    if errors:
        pd.DataFrame(errors).to_csv(RESULTS_DIR / "patching_control_errors.csv", index=False)
    else:
        stale_errors = RESULTS_DIR / "patching_control_errors.csv"
        if stale_errors.exists():
            stale_errors.unlink()
    if not frames:
        raise RuntimeError("No patching-control runs completed.")
    effects = pd.concat(frames, ignore_index=True, sort=False)
    summary = summarize(effects)
    boot = bootstrap_summary(effects, rng)
    save_table(effects, RESULTS_DIR / "patching_controls")
    save_table(summary, RESULTS_DIR / "patching_control_summary")
    save_table(boot, RESULTS_DIR / "patching_bootstrap")
    report = [
        "# Patching Controls Report",
        "",
        "Expanded patching was run on all 36 matched self/object pairs. Matched forward and reverse patches were run for `resid_pre`, `resid_post`, attention output, and MLP output. Additional random-source, same-condition, source-shuffle, mean-activation, and position controls were run for the primary `resid_post` stream to keep compute bounded while preserving the main causal control logic.",
        "",
        "The strongest allowed causal wording remains: residual states produced by self-directed contexts causally contribute to the measured inner-life-versus-object completion margin.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Bootstrap",
        "",
        boot.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    (REPORTS_DIR / "patching_controls_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {len(effects):,} patching-control rows.")


if __name__ == "__main__":
    main()

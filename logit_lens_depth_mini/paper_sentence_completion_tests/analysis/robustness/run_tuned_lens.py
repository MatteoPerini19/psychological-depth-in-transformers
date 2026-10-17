#!/usr/bin/env python3
"""Compare raw Logit Lens with official Tuned Lens readouts where available."""

from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens.nn.lenses import TunedLens

from common import CONFIG, OUTPUTS_DIR, REPORTS_DIR, RESULTS_DIR, ensure_dirs, layer_bands, mean_band, model_label, save_table


SCRIPT_01_PATH = Path(__file__).resolve().parents[2] / "scripts" / "01_run_binary_completion_probes_all_models.py"
spec = importlib.util.spec_from_file_location("binary_completion", SCRIPT_01_PATH)
binary_completion = importlib.util.module_from_spec(spec)
sys.modules["binary_completion"] = binary_completion
assert spec.loader is not None
spec.loader.exec_module(binary_completion)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    lens = TunedLens.from_model_and_pretrained(model, map_location="cpu")
    lens.to(device)
    lens.eval()
    return tokenizer, model, lens


def tuned_layer_logits_for_tokens(model, lens, tokenizer, stems: list[str], token_ids: list[int], device: torch.device, batch_size: int) -> np.ndarray:
    batches = []
    n_lens_layers = len(lens.layer_translators)
    for start in range(0, len(stems), batch_size):
        batch_stems = stems[start : start + batch_size]
        encoded = tokenizer(batch_stems, return_tensors="pt", padding=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        final_positions = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(len(batch_stems), device=device)
        per_layer = []
        for layer_index, hidden in enumerate(outputs.hidden_states):
            last_hidden = hidden[batch_index, final_positions, :]
            if layer_index < n_lens_layers:
                logits = lens(last_hidden, idx=layer_index)
            else:
                logits = binary_completion.output_logits_for_hidden(model, last_hidden)
            per_layer.append(logits[:, token_ids].detach().cpu().numpy())
        batches.append(np.stack(per_layer, axis=1))
    return np.concatenate(batches, axis=0)


def summarize_tuned(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_layer = binary_completion.summarize_by_layer(binary_completion.make_all_condition_metrics(metrics))
    summary = binary_completion.summarize_bands(binary_completion.make_all_condition_metrics(metrics))
    self_advantage = binary_completion.summarize_self_object_advantage(metrics)
    if not self_advantage.empty:
        summary = pd.concat([summary, self_advantage], ignore_index=True, sort=False)
    by_layer["readout"] = "tuned_lens"
    summary["readout"] = "tuned_lens"
    return by_layer, summary


def peak_bootstrap(metrics: pd.DataFrame, rng: np.random.Generator, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["model_name", "probe_name", "stem_condition"], sort=False):
        model_name, probe_name, condition = keys
        n_layers = int(group["layer_index"].max())
        pivot = group.pivot_table(index="stem_id", columns="layer_index", values="target_minus_contrast_logit_margin", aggfunc="mean")
        layer_cols = [col for col in sorted(pivot.columns) if col != 0]
        units = pivot.index.to_numpy()
        if len(units) < 2:
            continue
        peaks = []
        for _ in range(n_boot):
            sampled = rng.choice(units, size=len(units), replace=True)
            curve = pivot.loc[sampled, layer_cols].mean(axis=0)
            peaks.append(int(curve.idxmax()) / n_layers)
        rows.append(
            {
                "model_name": model_name,
                "model_label": model_label(model_name),
                "probe_name": probe_name,
                "stem_condition": condition,
                "readout": "tuned_lens",
                "peak_relative_depth_boot_mean": float(np.mean(peaks)),
                "peak_relative_depth_ci_low": float(np.quantile(peaks, 0.025)),
                "peak_relative_depth_ci_high": float(np.quantile(peaks, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    device = choose_device()
    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]) + 23)
    batch_size = 32
    all_metrics = []
    all_tokens = []
    status_rows = []

    selected_specs = binary_completion.PROBES
    for model_name in CONFIG["models"]:
        tokenizer = model = lens = None
        try:
            tokenizer, model, lens = prepare(model_name, device)
            status_rows.append(
                {
                    "model_name": model_name,
                    "model_label": model_label(model_name),
                    "tuned_lens_available": True,
                    "status": "official_checkpoint_loaded",
                    "details": str(lens.config),
                }
            )
        except Exception as exc:
            status_rows.append(
                {
                    "model_name": model_name,
                    "model_label": model_label(model_name),
                    "tuned_lens_available": False,
                    "status": "official_checkpoint_unavailable_or_failed",
                    "details": repr(exc),
                }
            )
            continue
        try:
            for probe in selected_specs:
                stems = binary_completion.load_stems(probe)
                token_checks = binary_completion.load_candidates(probe, tokenizer, model_name)
                token_checks["readout"] = "tuned_lens"
                all_tokens.append(token_checks)
                candidates = binary_completion.usable_candidates_for_metrics(token_checks)
                token_ids = sorted(int(token_id) for token_id in candidates["token_id"].unique())
                logits = tuned_layer_logits_for_tokens(model, lens, tokenizer, stems["stem"].tolist(), token_ids, device, batch_size)
                metrics, _word_metrics = binary_completion.build_probe_metrics(probe, stems, candidates, logits, token_ids, model_name)
                metrics["readout"] = "tuned_lens"
                all_metrics.append(metrics)
        finally:
            del lens
            del model
            del tokenizer
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    status = pd.DataFrame(status_rows)
    status.to_csv(RESULTS_DIR / "tuned_lens_status.csv", index=False)
    if all_tokens:
        pd.concat(all_tokens, ignore_index=True).to_csv(RESULTS_DIR / "tuned_lens_token_check.csv", index=False)
    if not all_metrics:
        report = [
            "# Tuned Lens Report",
            "",
            "No official Tuned Lens checkpoints could be loaded, so no tuned readout comparison was computed.",
            "",
            status.to_markdown(index=False),
        ]
        (REPORTS_DIR / "tuned_lens_report.md").write_text("\n".join(report), encoding="utf-8")
        print("No Tuned Lens runs completed.")
        return

    metrics = pd.concat(all_metrics, ignore_index=True)
    by_layer, summary = summarize_tuned(metrics)
    peak = peak_bootstrap(metrics, rng)
    raw_layer = pd.read_csv(OUTPUTS_DIR / "cross_model_sentence_completion_summary_by_layer.csv")
    raw_summary = pd.read_csv(OUTPUTS_DIR / "cross_model_sentence_completion_summary_by_model.csv")
    raw_layer = raw_layer[raw_layer["model_name"].isin(metrics["model_name"].unique())].copy()
    raw_summary = raw_summary[raw_summary["model_name"].isin(metrics["model_name"].unique())].copy()
    raw_layer["readout"] = "raw_logit_lens"
    raw_summary["readout"] = "raw_logit_lens"

    comparisons = []
    for keys, tuned_group in by_layer.groupby(["model_name", "probe_name", "stem_condition"], sort=False):
        model_name, probe_name, condition = keys
        raw_group = raw_layer[
            raw_layer["model_name"].eq(model_name)
            & raw_layer["probe_name"].eq(probe_name)
            & raw_layer["stem_condition"].eq(condition)
        ]
        if raw_group.empty:
            continue
        merged = tuned_group.merge(raw_group, on="layer_index", suffixes=("_tuned", "_raw"))
        rho = spearmanr(merged["mean_logit_margin_raw"], merged["mean_logit_margin_tuned"]).statistic
        tuned_row = summary[
            summary["model_name"].eq(model_name)
            & summary["probe_name"].eq(probe_name)
            & summary["stem_condition"].eq(condition)
        ]
        raw_row = raw_summary[
            raw_summary["model_name"].eq(model_name)
            & raw_summary["probe_name"].eq(probe_name)
            & raw_summary["stem_condition"].eq(condition)
        ]
        if tuned_row.empty or raw_row.empty:
            continue
        tuned_row = tuned_row.iloc[0]
        raw_row = raw_row.iloc[0]
        comparisons.append(
            {
                "model_name": model_name,
                "model_label": model_label(model_name),
                "probe_name": probe_name,
                "stem_condition": condition,
                "raw_late_minus_middle": raw_row["late_minus_middle"],
                "tuned_late_minus_middle": tuned_row["late_minus_middle"],
                "tuned_minus_raw_late_minus_middle": tuned_row["late_minus_middle"] - raw_row["late_minus_middle"],
                "raw_peak_relative_depth": raw_row["peak_relative_depth"],
                "tuned_peak_relative_depth": tuned_row["peak_relative_depth"],
                "spearman_layer_curve_raw_vs_tuned": rho,
                "raw_final_margin": raw_row["final_mean_margin"],
                "tuned_final_margin": tuned_row["final_mean_margin"],
            }
        )
    comparison = pd.DataFrame(comparisons)

    save_table(metrics, RESULTS_DIR / "tuned_lens_layerwise")
    save_table(summary, RESULTS_DIR / "tuned_lens_model_summary")
    save_table(comparison, RESULTS_DIR / "tuned_vs_raw_comparison")
    save_table(peak, RESULTS_DIR / "tuned_lens_peak_bootstrap")

    report = [
        "# Tuned Lens Report",
        "",
        "Official Tuned Lens checkpoints were used where available. No lens was trained on manuscript prompts. Models without official compatible checkpoints are reported as unavailable rather than approximated by an ad hoc translator.",
        "",
        "For Tuned Lens, layer indices `0..N-1` were read through the learned translators. The final block output `N` has no translator in the official lens object and is therefore scored with the model's normal final unembedding, matching final-layer behavior.",
        "",
        "## Checkpoint Status",
        "",
        status.to_markdown(index=False),
        "",
        "## Raw Versus Tuned Summary",
        "",
        comparison.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    (REPORTS_DIR / "tuned_lens_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote Tuned Lens comparison for {metrics['model_name'].nunique()} models.")


if __name__ == "__main__":
    main()

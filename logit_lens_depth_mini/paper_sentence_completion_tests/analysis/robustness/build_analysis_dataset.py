#!/usr/bin/env python3
"""Build canonical long-form robustness datasets from existing outputs."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from common import (
    CONFIG,
    OUTPUTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
    model_family,
    model_size,
    package_versions,
    save_table,
    write_json,
)


def standardize_aggregate() -> pd.DataFrame:
    source = OUTPUTS_DIR / "cross_model_sentence_completion_layer_metrics.csv"
    df = pd.read_csv(source, low_memory=False)
    out = df.rename(
        columns={
            "model_name": "model_id",
            "stem": "stem_text",
            "target_minus_contrast_logit_margin": "margin",
            "target_candidate_probability": "target_candidate_pool_probability",
            "contrast_candidate_probability": "contrast_candidate_pool_probability",
            "cova_dimension": "depth_dimension",
        }
    ).copy()
    out["model_family"] = out["model_id"].map(model_family)
    out["model_size"] = out["model_id"].map(model_size)
    out["probe"] = out["probe_name"]
    out["condition"] = out["stem_condition"]
    out["candidate_set"] = out["target_group_label"] + "_vs_" + out["contrast_group_label"]
    out["candidate_id"] = np.nan
    out["candidate_text"] = ""
    out["candidate_token_id"] = np.nan
    out["block_index"] = out["layer_index"].map(lambda x: np.nan if int(x) == 0 else int(x))
    out["depth_dimension"] = out.get("depth_dimension", "")
    out["lexical_control_flag"] = False
    out["intentional_mechanical_flag"] = False
    out["seed"] = CONFIG["seed"]
    out["source_file"] = str(source)
    columns = [
        "model_id",
        "model_family",
        "model_size",
        "probe",
        "prompt_id",
        "pair_id",
        "stem_text",
        "condition",
        "candidate_set",
        "candidate_id",
        "candidate_text",
        "candidate_token_id",
        "layer_index",
        "block_index",
        "relative_depth",
        "target_logsumexp",
        "contrast_logsumexp",
        "margin",
        "target_candidate_pool_probability",
        "contrast_candidate_pool_probability",
        "depth_dimension",
        "lexical_control_flag",
        "intentional_mechanical_flag",
        "seed",
        "source_file",
        "probe_name",
        "probe_label",
        "stem_template",
        "stem_family",
        "syntactic_frame",
        "lexical_type",
        "target_group_label",
        "contrast_group_label",
        "n_target_words",
        "n_contrast_words",
    ]
    out["prompt_id"] = out["stem_id"]
    return out[[column for column in columns if column in out.columns]]


def standardize_candidate_logits() -> pd.DataFrame:
    source = OUTPUTS_DIR / "cross_model_sentence_completion_word_metrics.csv"
    df = pd.read_csv(source, low_memory=False)
    out = df.rename(
        columns={
            "model_name": "model_id",
            "stem": "stem_text",
            "word": "candidate_text",
            "token_id": "candidate_token_id",
            "target_minus_contrast_logit_margin": "margin",
            "cova_dimension": "depth_dimension",
        }
    ).copy()
    out["model_family"] = out["model_id"].map(model_family)
    out["model_size"] = out["model_id"].map(model_size)
    out["probe"] = out["probe_name"]
    out["condition"] = out["stem_condition"]
    out["candidate_id"] = (
        out["probe_name"].astype(str)
        + "::"
        + out["syntactic_frame"].astype(str)
        + "::"
        + out["candidate_text"].astype(str)
    )
    out["candidate_set"] = out["target_group_label"] + "_vs_" + out["contrast_group_label"]
    out["block_index"] = out["layer_index"].map(lambda x: np.nan if int(x) == 0 else int(x))
    out["lexical_control_flag"] = False
    out["intentional_mechanical_flag"] = False
    out["seed"] = CONFIG["seed"]
    out["source_file"] = str(source)
    group_cols = ["model_id", "probe_name", "stem_id", "layer_index", "syntactic_frame"]
    out["candidate_pool_mean_logit"] = out.groupby(group_cols)["word_logit"].transform("mean")
    out["candidate_pool_centered_logit"] = out["word_logit"] - out["candidate_pool_mean_logit"]
    out["depth_dimension"] = out.get("depth_dimension", "")
    columns = [
        "model_id",
        "model_family",
        "model_size",
        "probe",
        "prompt_id",
        "pair_id",
        "stem_text",
        "condition",
        "candidate_set",
        "candidate_id",
        "candidate_text",
        "token_text",
        "candidate_token_id",
        "source_word_group",
        "analysis_group",
        "analysis_group_label",
        "layer_index",
        "block_index",
        "relative_depth",
        "word_logit",
        "candidate_pool_centered_logit",
        "candidate_set_probability",
        "depth_dimension",
        "lexical_control_flag",
        "intentional_mechanical_flag",
        "seed",
        "source_file",
        "probe_name",
        "probe_label",
        "stem_template",
        "stem_family",
        "syntactic_frame",
        "lexical_type",
        "role",
        "source_basis",
    ]
    out["prompt_id"] = out["stem_id"]
    return out[[column for column in columns if column in out.columns]]


def copy_existing_outputs_manifest() -> None:
    rows = []
    for path in sorted(OUTPUTS_DIR.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "suffix": path.suffix,
                    "size_bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "existing_outputs_manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    aggregate = standardize_aggregate()
    candidates = standardize_candidate_logits()
    save_table(aggregate, RESULTS_DIR / "analysis_dataset")
    save_table(candidates, RESULTS_DIR / "candidate_logits")
    copy_existing_outputs_manifest()
    write_json(package_versions(), RESULTS_DIR / "environment_info.json")
    config_copy = RESULTS_DIR / "config_used.yaml"
    shutil.copyfile(__import__("common").CONFIG_PATH, config_copy)
    print(f"Wrote {len(aggregate):,} aggregate rows and {len(candidates):,} candidate-logit rows.")


if __name__ == "__main__":
    main()

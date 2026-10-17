#!/usr/bin/env python3
"""Audit manuscript-facing robustness analyses without adding exploratory tests."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    CONFIG,
    DATA_DIR,
    PROJECT_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    aggregate_metrics,
    bootstrap_ci,
    ensure_dirs,
    layer_bands,
    package_versions,
    permutation_summary,
    save_table,
)
from run_hierarchical_models import self_advantage_unit_table
from run_layer_window_sensitivity import enumerate_windows


AUDIT_SEED = int(CONFIG["bootstrap_seed"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> dict:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return {"git_root": root, "git_commit": commit, "git_status_short": status}
    except Exception as exc:
        return {"git_root": None, "git_commit": None, "git_status_short": None, "git_error": repr(exc)}


def export_layer_window_specs() -> None:
    rows = []
    for n_layers in [6, 12]:
        middle, late = enumerate_windows(n_layers)
        for middle_window in middle:
            for late_window in late:
                if max(middle_window) >= min(late_window):
                    continue
                rows.append(
                    {
                        "n_transformer_layers": n_layers,
                        "middle_window": " ".join(map(str, middle_window)),
                        "late_window": " ".join(map(str, late_window)),
                        "middle_center_relative": float(np.mean(middle_window) / n_layers),
                        "late_center_relative": float(np.mean(late_window) / n_layers),
                    }
                )
    specs = pd.DataFrame(rows)
    save_table(specs[specs["n_transformer_layers"].eq(6)], RESULTS_DIR / "audit_layer_window_specs_6block")
    save_table(specs[specs["n_transformer_layers"].eq(12)], RESULTS_DIR / "audit_layer_window_specs_12block")

    code = inspect.getsource(enumerate_windows)
    (RESULTS_DIR / "audit_layer_window_generation_code.py").write_text(
        "# Exact function used to enumerate layer-window specifications.\n\n" + code,
        encoding="utf-8",
    )


def template_root(pair_id: str) -> str:
    return "_".join(str(pair_id).split("_")[1:])


def crossed_cluster_bootstrap(df: pd.DataFrame, unit_col: str, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    models = np.array(sorted(df["model_label"].unique()))
    units = np.array(sorted(df[unit_col].unique()))
    indexed = df.set_index(["model_label", unit_col])["late_minus_middle"]
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled = [float(indexed.loc[(model, unit)]) for model in sampled_models for unit in sampled_units]
        values[i] = float(np.mean(sampled))
    return values


def exact_sign_flip(values: np.ndarray, direction: str = "positive") -> dict:
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(values))), dtype=float)
    null = (signs * values[None, :]).mean(axis=1)
    return permutation_summary(float(np.mean(values)), null, direction=direction)


def rerun_template_level_tests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = aggregate_metrics()
    pair_units = self_advantage_unit_table(metrics).copy()
    pair_units["template_pair"] = pair_units["unit_id"].map(template_root)

    template_units = (
        pair_units.groupby(["model_id", "model_label", "template_pair"], as_index=False, sort=False)["late_minus_middle"]
        .mean()
        .rename(columns={"template_pair": "unit_id"})
    )
    template_units["probe_name"] = "self_object_advantage_template_level"
    save_table(template_units, RESULTS_DIR / "audit_template_level_unit_effects")

    pair_rng = np.random.default_rng(AUDIT_SEED)
    template_rng = np.random.default_rng(AUDIT_SEED)
    n_boot = int(CONFIG["candidate_bootstrap_resamples"])
    pair_boot = crossed_cluster_bootstrap(pair_units, "unit_id", pair_rng, n_boot)
    template_boot = crossed_cluster_bootstrap(template_units, "unit_id", template_rng, n_boot)

    pair_boot_df = pd.DataFrame({"bootstrap_index": np.arange(n_boot), "grand_mean": pair_boot})
    template_boot_df = pd.DataFrame({"bootstrap_index": np.arange(n_boot), "grand_mean": template_boot})
    save_table(pair_boot_df, RESULTS_DIR / "audit_pair_level_crossed_cluster_bootstrap")
    save_table(template_boot_df, RESULTS_DIR / "audit_template_level_crossed_cluster_bootstrap")

    bootstrap_summary = pd.DataFrame(
        [
            {
                "unit_level": "pair",
                "n_models": int(pair_units["model_label"].nunique()),
                "n_units": int(pair_units["unit_id"].nunique()),
                "estimate": float(pair_units["late_minus_middle"].mean()),
                "ci_low": bootstrap_ci(pair_boot)[0],
                "ci_high": bootstrap_ci(pair_boot)[1],
                "n_bootstrap": n_boot,
                "seed": AUDIT_SEED,
                "bootstrap_type": "crossed_model_by_unit_percentile",
            },
            {
                "unit_level": "template_pair",
                "n_models": int(template_units["model_label"].nunique()),
                "n_units": int(template_units["unit_id"].nunique()),
                "estimate": float(template_units["late_minus_middle"].mean()),
                "ci_low": bootstrap_ci(template_boot)[0],
                "ci_high": bootstrap_ci(template_boot)[1],
                "n_bootstrap": n_boot,
                "seed": AUDIT_SEED,
                "bootstrap_type": "crossed_model_by_unit_percentile",
            },
        ]
    )
    save_table(bootstrap_summary, RESULTS_DIR / "audit_crossed_cluster_bootstrap_summary")

    rows = []
    for model_label, group in template_units.groupby("model_label", sort=False):
        values = group.sort_values("unit_id")["late_minus_middle"].to_numpy(dtype=float)
        rows.append(
            {
                "model_label": model_label,
                "n_template_pairs": len(values),
                **exact_sign_flip(values),
            }
        )
    sign_flip = pd.DataFrame(rows)
    save_table(sign_flip, RESULTS_DIR / "audit_template_level_exact_sign_flip")
    return pair_units, template_units, bootstrap_summary, sign_flip


def fmt_p(value: float) -> str:
    if value <= 0.001:
        return "< .001"
    return f"{value:.3f}".replace("0.", ".")


def compare_float(table_name: str, row_id: str, field: str, reported: float, source: float, tol: float = 0.0005) -> dict:
    return {
        "table": table_name,
        "row_id": row_id,
        "field": field,
        "reported": reported,
        "source": source,
        "absolute_difference": abs(float(reported) - float(source)),
        "status": "PASS" if abs(float(reported) - float(source)) <= tol else "FAIL",
    }


def verify_robustness_tables() -> pd.DataFrame:
    checks: list[dict] = []
    perm = pd.read_csv(RESULTS_DIR / "permutation_results.csv")
    hier = pd.read_csv(RESULTS_DIR / "hierarchical_estimates.csv")
    layer = pd.read_csv(RESULTS_DIR / "layer_window_summary.csv")
    peak = pd.read_csv(RESULTS_DIR / "peak_bootstrap_summary.csv")
    cova_dim = pd.read_csv(PROJECT_DIR / "outputs" / "posthoc_cova_dimension_robustness.csv")

    expected_perm = {
        "gpt2": (4.971, 0.004, 0.910, 0.0001, 0.0001, 0.0005999, -1.745, 1.788, 5.459),
        "distilgpt2": (2.204, -0.004, 0.466, 0.0001, 0.0001, 0.0005999, -0.915, 0.898, 4.734),
        "pythia-14m": (-2.211, -0.000, 0.420, 1.0, 0.0001, 0.0005999, -0.830, 0.822, -5.260),
        "pythia-70m": (4.793, 0.003, 1.074, 0.0001, 0.0001, 0.0005999, -2.085, 2.079, 4.460),
        "pythia-70m-deduped": (11.802, -0.004, 2.096, 0.0001, 0.0001, 0.0005999, -4.071, 4.066, 5.633),
        "pythia-160m": (3.195, 0.002, 1.254, 0.004, 0.011, 0.011, -2.471, 2.427, 2.545),
    }
    perm_source = perm[perm["test_name"].eq("self_object_pair_sign_flip")]
    for model, values in expected_perm.items():
        src = perm_source[perm_source["model_label"].eq(model)].iloc[0]
        fields = [
            ("Observed Effect", "observed_statistic"),
            ("Null Mean", "null_mean"),
            ("Null SD", "null_sd"),
            ("One-Sided p", "empirical_p_one_sided"),
            ("Two-Sided p", "empirical_p_two_sided"),
            ("Holm p", "holm_p_two_sided_within_probe"),
            ("Null 95% low", "null_ci_low"),
            ("Null 95% high", "null_ci_high"),
            ("Standardized Effect", "standardized_null_effect"),
        ]
        for (display, source_col), reported in zip(fields, values, strict=True):
            checks.append(compare_float("Table 2 - matched-pair permutation", model, display, reported, src[source_col], tol=0.0015))

    expected_self_hier = {
        "naive_grand_mean": (4.126, np.nan, np.nan, 6, 36),
        "random_model_intercept_mixedlm": (4.126, 0.463, 7.789, 6, 36),
        "cluster_bootstrap_models_and_units": (4.126, 0.908, 7.669, 6, 36),
    }
    for estimator, values in expected_self_hier.items():
        src = hier[(hier["probe_name"].eq("self_object_advantage")) & (hier["estimator"].eq(estimator))].iloc[0]
        for field, reported, source_col in [
            ("Estimate", values[0], "estimate"),
            ("95% CI Low", values[1], "ci_low"),
            ("95% CI High", values[2], "ci_high"),
            ("Models", values[3], "n_models"),
            ("Pair Units", values[4], "n_units"),
        ]:
            if pd.isna(reported):
                ok = pd.isna(src[source_col])
                checks.append({"table": "Table 3 - self hierarchical", "row_id": estimator, "field": field, "reported": "NA", "source": src[source_col], "absolute_difference": np.nan, "status": "PASS" if ok else "FAIL"})
            else:
                checks.append(compare_float("Table 3 - self hierarchical", estimator, field, reported, src[source_col], tol=0.0015))

    expected_self_window = {
        "gpt2": (12, 101, 4.971, 1.000, 4.430, 2.066, 6.772, 3.163, 6.100),
        "distilgpt2": (6, 2, 2.204, 1.000, 2.740, 2.204, 3.276, 2.312, 3.169),
        "pythia-14m": (6, 2, -2.211, 0.000, -3.062, -3.914, -2.211, -3.743, -2.381),
        "pythia-70m": (6, 2, 4.793, 1.000, 5.797, 4.793, 6.801, 4.994, 6.600),
        "pythia-70m-deduped": (6, 2, 11.802, 1.000, 10.651, 9.499, 11.802, 9.730, 11.572),
        "pythia-160m": (12, 101, 3.195, 0.881, 2.382, -2.914, 6.610, -0.153, 4.499),
    }
    for model, values in expected_self_window.items():
        src = layer[(layer["probe_name"].eq("self_object_advantage")) & (layer["model_label"].eq(model))].iloc[0]
        for field, reported, source_col in [
            ("Blocks", values[0], "n_transformer_layers"),
            ("Window Specs", values[1], "n_window_specifications"),
            ("Primary Effect", values[2], "primary_effect"),
            ("Proportion Positive", values[3], "proportion_positive"),
            ("Median Effect", values[4], "median_effect"),
            ("Min", values[5], "min_effect"),
            ("Max", values[6], "max_effect"),
            ("10th Percentile", values[7], "q10_effect"),
            ("90th Percentile", values[8], "q90_effect"),
        ]:
            checks.append(compare_float("Table 4 - self layer windows", model, field, reported, src[source_col], tol=0.0015))

    expected_peak = {
        "gpt2": (0.833, 0.833, 0.833, 1.000, 0.000, 1.000),
        "distilgpt2": (0.833, 0.833, 0.833, 1.000, 0.000, 1.000),
        "pythia-14m": (0.169, 0.167, 0.167, 0.000, 0.003, 0.000),
        "pythia-70m": (0.833, 0.833, 0.833, 1.000, 0.000, 0.998),
        "pythia-70m-deduped": (0.667, 0.667, 0.667, 1.000, 0.000, 1.000),
        "pythia-160m": (0.848, 0.333, 0.917, 0.874, 0.000, 1.000),
    }
    for model, values in expected_peak.items():
        src = peak[(peak["probe_name"].eq("self_object_advantage")) & (peak["model_label"].eq(model))].iloc[0]
        for field, reported, source_col in [
            ("Mean Peak Relative Depth", values[0], "peak_relative_mean"),
            ("95% CI Low", values[1], "peak_relative_ci_low"),
            ("95% CI High", values[2], "peak_relative_ci_high"),
            ("P(Peak Late-Intermediate)", values[3], "prob_peak_late_intermediate"),
            ("P(Peak Final)", values[4], "prob_peak_final"),
            ("P(Late > Final)", values[5], "prob_late_exceeds_final"),
        ]:
            checks.append(compare_float("Table 5 - peak bootstrap", model, field, reported, src[source_col], tol=0.0015))

    expected_cova_hier = {
        "naive_grand_mean": (2.890, np.nan, np.nan, 6, 66),
        "random_model_intercept_mixedlm": (2.890, 0.471, 5.309, 6, 66),
        "cluster_bootstrap_models_and_units": (2.890, 0.870, 5.276, 6, 66),
    }
    for estimator, values in expected_cova_hier.items():
        src = hier[(hier["probe_name"].eq("cova_verb_noun")) & (hier["estimator"].eq(estimator))].iloc[0]
        for field, reported, source_col in [
            ("Estimate", values[0], "estimate"),
            ("95% CI Low", values[1], "ci_low"),
            ("95% CI High", values[2], "ci_high"),
            ("Models", values[3], "n_models"),
            ("Stem Units", values[4], "n_units"),
        ]:
            if pd.isna(reported):
                ok = pd.isna(src[source_col])
                checks.append({"table": "Table 6 - Cova hierarchical", "row_id": estimator, "field": field, "reported": "NA", "source": src[source_col], "absolute_difference": np.nan, "status": "PASS" if ok else "FAIL"})
            else:
                checks.append(compare_float("Table 6 - Cova hierarchical", estimator, field, reported, src[source_col], tol=0.0015))

    expected_cova_window = {
        "gpt2": (1.989, 1.000, 1.825, 1.109, 2.504),
        "distilgpt2": (0.969, 1.000, 0.993, 0.974, 1.012),
        "pythia-14m": (-0.501, 0.000, -0.927, -1.268, -0.587),
        "pythia-70m": (4.293, 1.000, 4.780, 4.391, 5.169),
        "pythia-70m-deduped": (8.138, 1.000, 7.278, 6.590, 7.966),
        "pythia-160m": (2.454, 0.921, 2.702, 0.294, 4.339),
    }
    for model, values in expected_cova_window.items():
        src = layer[(layer["probe_name"].eq("cova_verb_noun")) & (layer["model_label"].eq(model))].iloc[0]
        for field, reported, source_col in [
            ("Primary Effect", values[0], "primary_effect"),
            ("Proportion Positive", values[1], "proportion_positive"),
            ("Median Effect", values[2], "median_effect"),
            ("10th Percentile", values[3], "q10_effect"),
            ("90th Percentile", values[4], "q90_effect"),
        ]:
            checks.append(compare_float("Table 7 - Cova layer windows", model, field, reported, src[source_col], tol=0.0015))

    expected_cova_dim = {
        "cognitive stimulation/reflection": (1.97, 5, 6),
        "emotional impact/moved": (2.76, 6, 6),
        "epistemic progress/insight": (4.55, 6, 6),
        "hidden significance": (1.37, 5, 6),
        "important human condition": (2.13, 5, 6),
        "meaning/message": (2.28, 5, 6),
        "motivational change/transformation": (3.56, 6, 6),
        "personal relevance/values/identity": (1.18, 5, 6),
        "richness/layered content": (2.80, 5, 6),
    }
    dim_map = {
        "cognitive_stimulation_reflection": "cognitive stimulation/reflection",
        "emotional_impact_moved": "emotional impact/moved",
        "epistemic_progress_insight": "epistemic progress/insight",
        "hidden_significance": "hidden significance",
        "important_human_condition": "important human condition",
        "meaning_message": "meaning/message",
        "motivational_change_transformation": "motivational change/transformation",
        "personal_relevance_values_identity": "personal relevance/values/identity",
        "richness_layered_content": "richness/layered content",
    }
    cova_dim_summary = (
        cova_dim.assign(draft_dimension=cova_dim["cova_dimension"].map(dim_map))
        .groupby("draft_dimension", as_index=False, sort=False)
        .agg(
            mean_late_minus_middle=("late_minus_middle", "mean"),
            positive_models=("positive_late_shift", "sum"),
            models_tested=("model_label", "nunique"),
        )
    )
    for dim, values in expected_cova_dim.items():
        src = cova_dim_summary[cova_dim_summary["draft_dimension"].eq(dim)].iloc[0]
        for field, reported, source_col in [
            ("Mean late-minus-middle", values[0], "mean_late_minus_middle"),
            ("Positive models", values[1], "positive_models"),
            ("Models tested", values[2], "models_tested"),
        ]:
            checks.append(compare_float("Current draft Table 4 - Cova dimensions", dim, field, reported, src[source_col], tol=0.005))

    out = pd.DataFrame(checks)
    save_table(out, RESULTS_DIR / "audit_table_verification")
    return out


def write_manifest(table_checks: pd.DataFrame) -> dict:
    relevant_files = [
        Path("analysis/robustness/config.yaml"),
        Path("analysis/robustness/common.py"),
        Path("analysis/robustness/run_permutations.py"),
        Path("analysis/robustness/run_hierarchical_models.py"),
        Path("analysis/robustness/run_layer_window_sensitivity.py"),
        Path("analysis/robustness/audit_manuscript_robustness.py"),
        Path("data/self_object_stems.csv"),
        Path("results/robustness/analysis_dataset.csv"),
        Path("results/robustness/permutation_results.csv"),
        Path("results/robustness/hierarchical_estimates.csv"),
        Path("results/robustness/layer_window_summary.csv"),
        Path("results/robustness/layer_window_grid.csv"),
        Path("results/robustness/peak_bootstrap.csv"),
        Path("results/robustness/peak_bootstrap_summary.csv"),
        Path("outputs/posthoc_cova_dimension_robustness.csv"),
    ]
    hashes = {}
    for rel in relevant_files:
        path = PROJECT_DIR / rel
        if path.exists():
            hashes[str(rel)] = sha256(path)
    manifest = {
        "project": CONFIG["project_name"],
        "script_paths": [
            "analysis/robustness/build_analysis_dataset.py",
            "analysis/robustness/validate_reproduction.py",
            "analysis/robustness/run_permutations.py",
            "analysis/robustness/run_hierarchical_models.py",
            "analysis/robustness/run_layer_window_sensitivity.py",
            "analysis/robustness/audit_manuscript_robustness.py",
        ],
        "seeds": {
            "seed": CONFIG["seed"],
            "permutation_seed": CONFIG["permutation_seed"],
            "bootstrap_seed": CONFIG["bootstrap_seed"],
            "effective_peak_bootstrap_seed": int(CONFIG["bootstrap_seed"]) + 11,
            "patching_seed": CONFIG["patching_seed"],
        },
        "resamples": {
            "permutations": CONFIG["permutations"],
            "candidate_bootstrap_resamples": CONFIG["candidate_bootstrap_resamples"],
            "peak_bootstrap_resamples": CONFIG["peak_bootstrap_resamples"],
            "patching_bootstrap_resamples": CONFIG["patching_bootstrap_resamples"],
        },
        "packages": package_versions(),
        "hashes_sha256": hashes,
        "git": git_commit(),
        "table_verification": {
            "n_checks": int(len(table_checks)),
            "n_fail": int(table_checks["status"].eq("FAIL").sum()),
        },
    }
    (RESULTS_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_report(
    pair_units: pd.DataFrame,
    template_units: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    sign_flip: pd.DataFrame,
    table_checks: pd.DataFrame,
    manifest: dict,
) -> None:
    specs6 = pd.read_csv(RESULTS_DIR / "audit_layer_window_specs_6block.csv")
    specs12 = pd.read_csv(RESULTS_DIR / "audit_layer_window_specs_12block.csv")
    fail_count = int(table_checks["status"].eq("FAIL").sum())
    report = [
        "# Robustness Manuscript Audit",
        "",
        "This audit checks only robustness analyses already selected for the manuscript-facing sentence-completion package. It does not add exploratory analyses.",
        "",
        "## 1. Peak-Location Bootstrap",
        "",
        f"- Resamples: {CONFIG['peak_bootstrap_resamples']:,}.",
        f"- Config bootstrap seed: {CONFIG['bootstrap_seed']}.",
        f"- Effective peak-bootstrap seed: {int(CONFIG['bootstrap_seed']) + 11}, because `run_layer_window_sensitivity.py` initializes the generator as `bootstrap_seed + 11`.",
        "- Confidence interval method: percentile bootstrap, 2.5th and 97.5th percentiles via `np.quantile`.",
        "- Peak tie-breaking rule: `pandas.Series.idxmax()` over sorted non-embedding layer columns, so exact ties are resolved in favor of the earliest layer among tied maxima.",
        "- Embedding layer excluded from peak candidates.",
        "",
        "## 2. Layer-Window Specifications",
        "",
        f"- 6-block models: {len(specs6)} specifications exported to `results/robustness/audit_layer_window_specs_6block.csv`.",
        f"- 12-block models: {len(specs12)} specifications exported to `results/robustness/audit_layer_window_specs_12block.csv`.",
        "- Exact generation code exported to `results/robustness/audit_layer_window_generation_code.py`.",
        "",
        "## 3. Cluster Bootstrap Structure",
        "",
        "The existing manuscript-facing cluster bootstrap sampled models with replacement and then sampled units independently within each sampled model. That procedure is documented and reproducible, but it treats units as nested within model draws rather than preserving the complete crossed model-by-unit structure.",
        "",
        "For this audit, I reran a crossed bootstrap that samples model labels and unit labels independently, then includes the full sampled model-by-unit grid with multiplicity. This preserves the crossed structure because the same unit identifiers are used across all sampled models.",
        "",
        bootstrap_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## 4. Prompt-Pair Template Dependence",
        "",
        f"The self/object materials contain {pair_units['unit_id'].nunique()} name-level prompt pairs but only {template_units['unit_id'].nunique()} independent template-pair families. Each template-pair family is repeated across six names.",
        "",
        "Therefore, the highest defensible independent stimulus level is the template-pair level. I aggregated pair deltas over names within each model-template cell and reran exact sign-flip tests over the six template-pair units per model.",
        "",
        sign_flip.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## 5. Table Verification",
        "",
        f"Verification checks performed: {len(table_checks)}.",
        f"Failed checks: {fail_count}.",
        "",
        "Verified source tables include `permutation_results.csv`, `hierarchical_estimates.csv`, `layer_window_summary.csv`, `peak_bootstrap_summary.csv`, and `posthoc_cova_dimension_robustness.csv`.",
        "",
        "## 6. Reproducibility Manifest",
        "",
        "Manifest exported to `results/robustness/reproducibility_manifest.json`.",
        f"Git commit hash: {manifest['git'].get('git_commit') or 'not available; folder is not inside an active Git repository'}",
        "",
        "## Output Files",
        "",
        "- `results/robustness/audit_layer_window_specs_6block.csv`",
        "- `results/robustness/audit_layer_window_specs_12block.csv`",
        "- `results/robustness/audit_layer_window_generation_code.py`",
        "- `results/robustness/audit_pair_level_crossed_cluster_bootstrap.csv`",
        "- `results/robustness/audit_template_level_crossed_cluster_bootstrap.csv`",
        "- `results/robustness/audit_crossed_cluster_bootstrap_summary.csv`",
        "- `results/robustness/audit_template_level_unit_effects.csv`",
        "- `results/robustness/audit_template_level_exact_sign_flip.csv`",
        "- `results/robustness/audit_table_verification.csv`",
        "- `results/robustness/reproducibility_manifest.json`",
        "",
    ]
    (REPORTS_DIR / "robustness_manuscript_audit.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    export_layer_window_specs()
    pair_units, template_units, bootstrap_summary, sign_flip = rerun_template_level_tests()
    table_checks = verify_robustness_tables()
    manifest = write_manifest(table_checks)
    write_report(pair_units, template_units, bootstrap_summary, sign_flip, table_checks, manifest)
    failures = int(table_checks["status"].eq("FAIL").sum())
    print(f"Audit complete. Table verification failures: {failures}.")


if __name__ == "__main__":
    sys.exit(main())

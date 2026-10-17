#!/usr/bin/env python3
"""Shared utilities for the robustness-analysis pipeline."""

from __future__ import annotations

import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONFIG = load_config()
DATA_DIR = PROJECT_DIR / CONFIG["paths"]["data_dir"]
OUTPUTS_DIR = PROJECT_DIR / CONFIG["paths"]["existing_outputs_dir"]
RESULTS_DIR = PROJECT_DIR / CONFIG["paths"]["robustness_results_dir"]
FIGURES_DIR = PROJECT_DIR / CONFIG["paths"]["robustness_figures_dir"]
REPORTS_DIR = PROJECT_DIR / CONFIG["paths"]["reports_dir"]
MIDDLE_MIN_RELATIVE = float(CONFIG["middle_min_relative"])
MIDDLE_MAX_RELATIVE = float(CONFIG["middle_max_relative"])
LATE_MIN_RELATIVE = float(CONFIG["late_min_relative"])


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def model_label(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.split("/")[-1])


def model_family(model_name: str) -> str:
    if "pythia" in model_name.lower():
        return "pythia"
    if "gpt2" in model_name.lower():
        return "gpt2"
    return "other"


def model_size(model_name: str) -> str:
    lower = model_name.lower()
    for size in ["14m", "70m", "160m"]:
        if size in lower:
            return size.upper()
    if lower == "gpt2":
        return "124M"
    if lower == "distilgpt2":
        return "82M"
    return ""


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def save_table(df: pd.DataFrame, path_base: Path, *, index: bool = False) -> None:
    """Save a table to CSV and, when available, Parquet."""
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=index)
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=index)
    except Exception as exc:  # pragma: no cover - documented in report.
        warning = path_base.with_suffix(".parquet.warning.txt")
        warning.write_text(f"Parquet export failed: {exc!r}\n", encoding="utf-8")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def layer_bands(max_layer: int) -> dict[str, list[int]]:
    layers = list(range(max_layer + 1))
    middle = [
        layer
        for layer in layers
        if layer != 0 and MIDDLE_MIN_RELATIVE <= layer / max_layer <= MIDDLE_MAX_RELATIVE
    ]
    late = [layer for layer in layers if LATE_MIN_RELATIVE < layer / max_layer < 1.0]
    if not middle:
        middle = [max(1, int(round(max_layer * 0.4)))]
    if not late:
        late = [max(1, max_layer - 1)]
    return {"middle": middle, "late": late, "final": [max_layer]}


def layer_region(layer_index: int, max_layer: int) -> str:
    if layer_index == 0:
        return "embedding"
    if layer_index == max_layer:
        return "final"
    rel = layer_index / max_layer
    if MIDDLE_MIN_RELATIVE <= rel <= MIDDLE_MAX_RELATIVE:
        return "middle"
    if LATE_MIN_RELATIVE < rel < 1.0:
        return "late_intermediate"
    return "early"


def block_index(layer_index: int) -> int | float:
    return np.nan if layer_index == 0 else int(layer_index)


def logsumexp_np(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_value = np.nanmax(values, axis=axis, keepdims=True)
    out = np.log(np.nansum(np.exp(values - max_value), axis=axis, keepdims=True)) + max_value
    if axis is None:
        return np.squeeze(out)
    return np.squeeze(out, axis=axis)


def softmax_np(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def mean_band(table: pd.DataFrame, layers: Iterable[int]) -> pd.Series:
    available = [layer for layer in layers if layer in table.columns]
    if not available:
        return pd.Series(np.nan, index=table.index)
    return table[available].mean(axis=1)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def bh_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    prev = 1.0
    for reverse_rank, idx in enumerate(order[::-1], start=1):
        rank = m - reverse_rank + 1
        value = min(prev, p[idx] * m / rank)
        adjusted[idx] = min(1.0, value)
        prev = value
    return adjusted.tolist()


def permutation_summary(observed: float, null_values: np.ndarray, direction: str = "positive") -> dict:
    null_values = np.asarray(null_values, dtype=float)
    b = int(len(null_values))
    two_sided = (1 + np.sum(np.abs(null_values) >= abs(observed))) / (b + 1)
    if direction == "positive":
        one_sided = (1 + np.sum(null_values >= observed)) / (b + 1)
    else:
        one_sided = (1 + np.sum(null_values <= observed)) / (b + 1)
    pct = float((np.sum(null_values < observed) + 0.5 * np.sum(null_values == observed)) / b)
    sd = float(null_values.std(ddof=1))
    return {
        "observed_statistic": float(observed),
        "null_mean": float(null_values.mean()),
        "null_sd": sd,
        "empirical_p_two_sided": float(two_sided),
        "empirical_p_one_sided": float(one_sided),
        "null_percentile_observed": pct,
        "standardized_null_effect": float((observed - null_values.mean()) / sd) if sd > 0 else np.nan,
        "monte_carlo_se_two_sided": float(math.sqrt(two_sided * (1 - two_sided) / (b + 1))),
        "null_ci_low": float(np.quantile(null_values, 0.025)),
        "null_ci_high": float(np.quantile(null_values, 0.975)),
        "n_permutations": b,
    }


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def candidate_level_logits() -> pd.DataFrame:
    path = RESULTS_DIR / "candidate_logits.parquet"
    if path.exists():
        return pd.read_parquet(path)
    csv_path = RESULTS_DIR / "candidate_logits.csv"
    if csv_path.exists():
        return safe_read_csv(csv_path)
    return safe_read_csv(OUTPUTS_DIR / "cross_model_sentence_completion_word_metrics.csv")


def aggregate_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "analysis_dataset.parquet"
    if path.exists():
        return pd.read_parquet(path)
    csv_path = RESULTS_DIR / "analysis_dataset.csv"
    if csv_path.exists():
        return safe_read_csv(csv_path)
    return safe_read_csv(OUTPUTS_DIR / "cross_model_sentence_completion_layer_metrics.csv")


def package_versions() -> dict:
    names = [
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "matplotlib",
        "pyarrow",
        "scipy",
        "statsmodels",
        "tuned_lens",
    ]
    versions = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            versions[name] = f"unavailable: {exc!r}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def report_path(name: str) -> Path:
    return REPORTS_DIR / name


def figure_path(name: str) -> Path:
    return FIGURES_DIR / name


def result_path(name: str) -> Path:
    return RESULTS_DIR / name

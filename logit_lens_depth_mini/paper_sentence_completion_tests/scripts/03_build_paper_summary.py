#!/usr/bin/env python3
"""Build paper-facing summary tables for the sentence-completion test package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


def fmt(value: float | int | str | bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) < 0.001 and float(value) != 0:
            return f"{float(value):.2e}"
        return f"{float(value):.3f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(fmt)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, divider, *rows])


def strength_label(n_positive: int, n_significant_positive: int, n_models: int) -> str:
    if n_positive == n_models and n_significant_positive >= n_models - 1:
        return "strong"
    if n_positive >= n_models - 1 and n_significant_positive >= max(1, n_models - 2):
        return "mostly supported"
    if n_positive >= n_models // 2:
        return "partial"
    return "weak/mixed"


def binary_rows(summary: pd.DataFrame) -> list[dict]:
    rows = []
    specs = [
        (
            "Original minimal stems: reflective > practical",
            "reflective_practical",
            "all_stems",
            "Reflective verbs should gain relative support in late-intermediate layers.",
        ),
        (
            "Self-directed advantage",
            "self_object_advantage",
            "self_minus_object_stems",
            "Self-directed stems should show a larger inner-life noun advantage than object-directed stems.",
        ),
        (
            "Cova-derived verb/noun depth probe",
            "cova_verb_noun",
            "all_stems",
            "Cova-derived depth-positive verbs/nouns should gain support over practical controls.",
        ),
    ]
    for analysis_name, probe_name, condition, expectation in specs:
        selected = summary[
            summary["probe_name"].eq(probe_name) & summary["stem_condition"].eq(condition)
        ].copy()
        n_models = int(selected["model_label"].nunique())
        positive = selected["late_minus_middle"].gt(0)
        significant_positive = positive & selected["late_minus_middle_p_two_sided"].lt(0.05)
        late_above_final = selected["late_mean_margin"].gt(selected["final_mean_margin"])
        rows.append(
            {
                "analysis": analysis_name,
                "expected_result": expectation,
                "models_positive": f"{int(positive.sum())}/{n_models}",
                "models_significant_positive": f"{int(significant_positive.sum())}/{n_models}",
                "models_late_above_final": f"{int(late_above_final.sum())}/{n_models}",
                "mean_late_minus_middle": float(selected["late_minus_middle"].mean()),
                "median_late_minus_middle": float(selected["late_minus_middle"].median()),
                "result": strength_label(int(positive.sum()), int(significant_positive.sum()), n_models),
            }
        )
    return rows


def abstraction_rows(summary: pd.DataFrame) -> list[dict]:
    rows = []
    for lexical_type in ["verb", "noun"]:
        selected = summary[summary["lexical_type"].eq(lexical_type)].copy()
        n_models = int(selected["model_label"].nunique())
        positive_expected = selected["expected_late_minus_middle"].gt(0)
        significant_expected = positive_expected & selected["expected_late_minus_middle_p_two_sided"].lt(0.05)
        positive_margin = selected["margin_late_minus_middle"].gt(0)
        rows.append(
            {
                "analysis": f"Abstraction gradient: {lexical_type}s",
                "expected_result": (
                    "Higher layers should shift probability mass from concrete/social candidates "
                    "toward reflective/identity-related candidates."
                ),
                "models_positive": f"{int(positive_expected.sum())}/{n_models}",
                "models_significant_positive": f"{int(significant_expected.sum())}/{n_models}",
                "models_late_above_final": "",
                "mean_late_minus_middle": float(selected["expected_late_minus_middle"].mean()),
                "median_late_minus_middle": float(selected["expected_late_minus_middle"].median()),
                "result": strength_label(int(positive_expected.sum()), int(significant_expected.sum()), n_models),
                "margin_positive_models": f"{int(positive_margin.sum())}/{n_models}",
            }
        )
    return rows


def main() -> None:
    binary = pd.read_csv(OUTPUT_DIR / "cross_model_sentence_completion_summary_by_model.csv")
    abstraction = pd.read_csv(OUTPUT_DIR / "abstraction_gradient_cross_model_band_summary.csv")
    rows = binary_rows(binary) + abstraction_rows(abstraction)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "paper_test_confirmation_table.csv", index=False)
    md_table = markdown_table(table)
    (OUTPUT_DIR / "paper_test_confirmation_table.md").write_text(md_table + "\n", encoding="utf-8")

    note = [
        "# Paper Sentence-Completion Test Package Summary",
        "",
        "This folder contains the analyses selected for a possible paper focused on sentence-completion Logit Lens probes. The package excludes the archived deep-vs-superficial label-classification route.",
        "",
        "## Included Analyses",
        "",
        "- Original minimal stems: reflective/practical verb candidates after simple stems such as `John wants to`.",
        "- Self-directed vs object-directed stems: inner-life nouns after self-directed stems compared with object/logistic nouns after object-directed stems.",
        "- Cova-derived verb/noun probe: theoretically grounded depth-positive candidates compared with practical controls.",
        "- Abstraction-gradient probe: concrete/practical, social/emotional, reflective/abstract, and existential/identity candidate groups, split by verbs and nouns.",
        "",
        "## Confirmation Table",
        "",
        md_table,
        "",
        "## Reading Rule",
        "",
        "The main confirmatory quantity is the late-minus-middle contrast. Positive values indicate that the target depth-related candidate group gained relative support in late-intermediate layers compared with middle layers. Stronger paper claims should focus on analyses marked `strong` or `mostly supported`; `partial` and `weak/mixed` analyses are better treated as specificity or boundary-condition results.",
        "",
    ]
    (PROJECT_DIR / "Paper Sentence Completion Test Package Summary.md").write_text(
        "\n".join(note),
        encoding="utf-8",
    )
    print(md_table)
    print(f"\nWrote {OUTPUT_DIR / 'paper_test_confirmation_table.csv'}")
    print(f"Wrote {PROJECT_DIR / 'Paper Sentence Completion Test Package Summary.md'}")


if __name__ == "__main__":
    main()

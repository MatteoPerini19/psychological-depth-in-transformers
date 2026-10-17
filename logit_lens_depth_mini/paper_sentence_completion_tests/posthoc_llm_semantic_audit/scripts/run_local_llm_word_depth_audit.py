#!/usr/bin/env python3
"""Post hoc local-LLM semantic audit for paper-facing candidate words.

This script rates the analyzed sentence-completion candidate words with a
local Ollama model. The prompt is blinded to experimental labels: it includes
only the word and neutral part-of-speech hints. Source-group metadata are
joined back after rating so downstream validation scripts can test the
manuscript-facing contrasts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "source_word_lists"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ratings"

MODEL_NAME = "gpt-oss:20b"
MODEL_PREFIX = "gpt_oss_20b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SOURCE_FILES = {
    "simple_word_groups": DATA_DIR / "simple_word_groups.csv",
    "self_object_word_groups": DATA_DIR / "self_object_word_groups.csv",
    "cova_depth_completion_words_single_token": DATA_DIR / "cova_depth_completion_words_single_token.csv",
    "abstraction_gradient_split_word_groups": DATA_DIR / "abstraction_gradient_split_word_groups.csv",
}

RAW_SCORE_COLUMNS = [
    f"{MODEL_PREFIX}_psychological_depth_raw",
    f"{MODEL_PREFIX}_self_reflective_relevance_raw",
    f"{MODEL_PREFIX}_existential_meaning_raw",
    f"{MODEL_PREFIX}_reflective_cognitive_process_raw",
    f"{MODEL_PREFIX}_personal_emotional_salience_raw",
    f"{MODEL_PREFIX}_concrete_observable_reversed_raw",
    f"{MODEL_PREFIX}_logistical_transactional_reversed_raw",
    f"{MODEL_PREFIX}_impersonal_neutral_reversed_raw",
]

RECODED_COLUMNS = [
    f"{MODEL_PREFIX}_concrete_observable_depth_keyed",
    f"{MODEL_PREFIX}_logistical_transactional_depth_keyed",
    f"{MODEL_PREFIX}_impersonal_neutral_depth_keyed",
]

COMPOSITE_COLUMN = f"{MODEL_PREFIX}_psychological_depth_composite"

ITEM_DEFINITIONS = [
    {
        "column": f"{MODEL_PREFIX}_psychological_depth_raw",
        "prompt_key": "psychological_depth",
        "reverse_keyed": False,
        "definition": "High scores mean the word evokes psychological depth, inner life, self-reflection, or meaning-laden mental content.",
    },
    {
        "column": f"{MODEL_PREFIX}_self_reflective_relevance_raw",
        "prompt_key": "self_reflective_relevance",
        "reverse_keyed": False,
        "definition": "High scores mean the word is naturally relevant to thinking about oneself, one's motives, feelings, identity, or life direction.",
    },
    {
        "column": f"{MODEL_PREFIX}_existential_meaning_raw",
        "prompt_key": "existential_meaning",
        "reverse_keyed": False,
        "definition": "High scores mean the word evokes identity, purpose, values, mortality, life meaning, or existential significance.",
    },
    {
        "column": f"{MODEL_PREFIX}_reflective_cognitive_process_raw",
        "prompt_key": "reflective_cognitive_process",
        "reverse_keyed": False,
        "definition": "High scores mean the word suggests reflection, insight, understanding, reconsideration, or deeper interpretation.",
    },
    {
        "column": f"{MODEL_PREFIX}_personal_emotional_salience_raw",
        "prompt_key": "personal_emotional_salience",
        "reverse_keyed": False,
        "definition": "High scores mean the word feels personally or emotionally significant rather than emotionally flat.",
    },
    {
        "column": f"{MODEL_PREFIX}_concrete_observable_reversed_raw",
        "prompt_key": "concrete_observable",
        "reverse_keyed": True,
        "definition": "High raw scores mean the word is concrete, externally observable, practical, or surface-level. This item is reverse-keyed for psychological depth.",
    },
    {
        "column": f"{MODEL_PREFIX}_logistical_transactional_reversed_raw",
        "prompt_key": "logistical_transactional",
        "reverse_keyed": True,
        "definition": "High raw scores mean the word concerns mundane logistics, transactions, tools, objects, administration, or routine tasks. This item is reverse-keyed for psychological depth.",
    },
    {
        "column": f"{MODEL_PREFIX}_impersonal_neutral_reversed_raw",
        "prompt_key": "impersonal_neutral",
        "reverse_keyed": True,
        "definition": "High raw scores mean the word is impersonal, generic, emotionally neutral, or detached from inner experience. This item is reverse-keyed for psychological depth.",
    },
]


def prefix_from_model_name(model: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]+", "_", model.strip().lower()).strip("_")
    return prefix or "ollama_model"


def build_score_schema(prefix: str) -> tuple[list[str], list[str], str, list[dict[str, Any]]]:
    raw_score_columns = [
        f"{prefix}_psychological_depth_raw",
        f"{prefix}_self_reflective_relevance_raw",
        f"{prefix}_existential_meaning_raw",
        f"{prefix}_reflective_cognitive_process_raw",
        f"{prefix}_personal_emotional_salience_raw",
        f"{prefix}_concrete_observable_reversed_raw",
        f"{prefix}_logistical_transactional_reversed_raw",
        f"{prefix}_impersonal_neutral_reversed_raw",
    ]
    recoded_columns = [
        f"{prefix}_concrete_observable_depth_keyed",
        f"{prefix}_logistical_transactional_depth_keyed",
        f"{prefix}_impersonal_neutral_depth_keyed",
    ]
    composite_column = f"{prefix}_psychological_depth_composite"
    item_definitions = [
        {
            "column": f"{prefix}_psychological_depth_raw",
            "prompt_key": "psychological_depth",
            "reverse_keyed": False,
            "definition": "High scores mean the word evokes psychological depth, inner life, self-reflection, or meaning-laden mental content.",
        },
        {
            "column": f"{prefix}_self_reflective_relevance_raw",
            "prompt_key": "self_reflective_relevance",
            "reverse_keyed": False,
            "definition": "High scores mean the word is naturally relevant to thinking about oneself, one's motives, feelings, identity, or life direction.",
        },
        {
            "column": f"{prefix}_existential_meaning_raw",
            "prompt_key": "existential_meaning",
            "reverse_keyed": False,
            "definition": "High scores mean the word evokes identity, purpose, values, mortality, life meaning, or existential significance.",
        },
        {
            "column": f"{prefix}_reflective_cognitive_process_raw",
            "prompt_key": "reflective_cognitive_process",
            "reverse_keyed": False,
            "definition": "High scores mean the word suggests reflection, insight, understanding, reconsideration, or deeper interpretation.",
        },
        {
            "column": f"{prefix}_personal_emotional_salience_raw",
            "prompt_key": "personal_emotional_salience",
            "reverse_keyed": False,
            "definition": "High scores mean the word feels personally or emotionally significant rather than emotionally flat.",
        },
        {
            "column": f"{prefix}_concrete_observable_reversed_raw",
            "prompt_key": "concrete_observable",
            "reverse_keyed": True,
            "definition": "High raw scores mean the word is concrete, externally observable, practical, or surface-level. This item is reverse-keyed for psychological depth.",
        },
        {
            "column": f"{prefix}_logistical_transactional_reversed_raw",
            "prompt_key": "logistical_transactional",
            "reverse_keyed": True,
            "definition": "High raw scores mean the word concerns mundane logistics, transactions, tools, objects, administration, or routine tasks. This item is reverse-keyed for psychological depth.",
        },
        {
            "column": f"{prefix}_impersonal_neutral_reversed_raw",
            "prompt_key": "impersonal_neutral",
            "reverse_keyed": True,
            "definition": "High raw scores mean the word is impersonal, generic, emotionally neutral, or detached from inner experience. This item is reverse-keyed for psychological depth.",
        },
    ]
    return raw_score_columns, recoded_columns, composite_column, item_definitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-prefix", default="", help="Column/file prefix. Defaults to a sanitized model name.")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse existing raw response JSONL if present.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def join_values(values: set[str]) -> str:
    cleaned = sorted(str(value) for value in values if value not in {"", "None", "nan"} and value is not None)
    return ";".join(cleaned)


def load_analyzed_word_metadata() -> list[dict[str, str]]:
    rows_by_word: dict[str, dict[str, Any]] = {}

    def ensure_word(word: str) -> dict[str, Any]:
        key = word.strip().lower()
        if key not in rows_by_word:
            rows_by_word[key] = {
                "word": key,
                "source_tables": set(),
                "source_memberships": set(),
                "roles": set(),
                "lexical_types": set(),
                "syntactic_frames": set(),
                "cova_dimensions": set(),
                "cova_polarities": set(),
                "abstraction_levels": set(),
                "simple_word_group": "",
                "simple_claim_group": "",
                "self_object_word_group": "",
                "self_object_claim_group": "",
                "cova_word_groups": set(),
                "cova_claim_group": "",
                "abstraction_word_groups": set(),
                "abstraction_min_level": "",
                "abstraction_max_level": "",
                "analysis_claim_groups": set(),
            }
        return rows_by_word[key]

    for row in read_csv(SOURCE_FILES["simple_word_groups"]):
        word_row = ensure_word(row["word"])
        word_row["source_tables"].add("simple_word_groups")
        word_row["source_memberships"].add(f"simple_word_groups:{row['word_group']}")
        word_row["roles"].add(row.get("role", ""))
        word_row["lexical_types"].add("verb")
        word_row["syntactic_frames"].add("verb_after_to")
        word_row["simple_word_group"] = row["word_group"]
        claim = "depth_related" if row["word_group"] == "deep" else "control"
        word_row["simple_claim_group"] = claim
        word_row["analysis_claim_groups"].add(claim)

    for row in read_csv(SOURCE_FILES["self_object_word_groups"]):
        word_row = ensure_word(row["word"])
        word_row["source_tables"].add("self_object_word_groups")
        word_row["source_memberships"].add(f"self_object_word_groups:{row['word_group']}")
        word_row["roles"].add(row.get("role", ""))
        word_row["lexical_types"].add("noun")
        word_row["syntactic_frames"].add("noun_after_possessive_or_article")
        word_row["self_object_word_group"] = row["word_group"]
        claim = "depth_related" if row["word_group"] == "inner" else "control"
        word_row["self_object_claim_group"] = claim
        word_row["analysis_claim_groups"].add(claim)

    for row in read_csv(SOURCE_FILES["cova_depth_completion_words_single_token"]):
        if row.get("lexical_type") not in {"verb", "noun"}:
            continue
        if row.get("word_group") not in {"depth", "surface", "neutral_control"}:
            continue
        word_row = ensure_word(row["word"])
        word_row["source_tables"].add("cova_depth_completion_words_single_token")
        word_row["source_memberships"].add(f"cova_depth_completion_words_single_token:{row['word_group']}")
        word_row["roles"].add(row.get("polarity", ""))
        word_row["lexical_types"].add(row.get("lexical_type", ""))
        word_row["syntactic_frames"].add(row.get("syntactic_frame", ""))
        word_row["cova_dimensions"].add(row.get("cova_dimension", ""))
        word_row["cova_polarities"].add(row.get("polarity", ""))
        word_row["cova_word_groups"].add(row.get("word_group", ""))
        claim = "depth_related" if row["word_group"] == "depth" else "control"
        if word_row["cova_claim_group"] not in {"depth_related"}:
            word_row["cova_claim_group"] = claim
        word_row["analysis_claim_groups"].add(claim)

    for row in read_csv(SOURCE_FILES["abstraction_gradient_split_word_groups"]):
        word_row = ensure_word(row["word"])
        level = int(row["abstraction_level"])
        word_row["source_tables"].add("abstraction_gradient_split_word_groups")
        word_row["source_memberships"].add(f"abstraction_gradient_split_word_groups:{row['word_group']}")
        word_row["roles"].add(row.get("role", ""))
        word_row["lexical_types"].add(row.get("lexical_type", ""))
        word_row["syntactic_frames"].add(row.get("syntactic_frame", ""))
        word_row["abstraction_levels"].add(level)
        word_row["abstraction_word_groups"].add(row.get("word_group", ""))

        if level >= 3:
            word_row["analysis_claim_groups"].add("depth_related")
        elif level == 1:
            word_row["analysis_claim_groups"].add("control")
        else:
            word_row["analysis_claim_groups"].add("intermediate")

    output_rows = []
    for word in sorted(rows_by_word):
        raw = rows_by_word[word]
        levels = sorted(raw["abstraction_levels"])
        claim_groups = set(raw["analysis_claim_groups"])
        if "depth_related" in claim_groups and "control" not in claim_groups:
            overall_claim = "depth_related"
        elif "control" in claim_groups and "depth_related" not in claim_groups:
            overall_claim = "control"
        elif "depth_related" in claim_groups and "control" in claim_groups:
            overall_claim = "mixed"
        elif "intermediate" in claim_groups:
            overall_claim = "intermediate"
        else:
            overall_claim = ""

        output_rows.append(
            {
                "word": raw["word"],
                "source_tables": join_values(raw["source_tables"]),
                "source_memberships": join_values(raw["source_memberships"]),
                "roles": join_values(raw["roles"]),
                "lexical_types": join_values(raw["lexical_types"]),
                "syntactic_frames": join_values(raw["syntactic_frames"]),
                "simple_word_group": raw["simple_word_group"],
                "simple_claim_group": raw["simple_claim_group"],
                "self_object_word_group": raw["self_object_word_group"],
                "self_object_claim_group": raw["self_object_claim_group"],
                "cova_word_groups": join_values(raw["cova_word_groups"]),
                "cova_claim_group": raw["cova_claim_group"],
                "cova_dimensions": join_values(raw["cova_dimensions"]),
                "cova_polarities": join_values(raw["cova_polarities"]),
                "abstraction_word_groups": join_values(raw["abstraction_word_groups"]),
                "abstraction_levels": ";".join(str(level) for level in levels),
                "abstraction_min_level": str(min(levels)) if levels else "",
                "abstraction_max_level": str(max(levels)) if levels else "",
                "overall_claim_group": overall_claim,
                "any_depth_related_claim": "true" if "depth_related" in claim_groups else "false",
                "any_control_claim": "true" if "control" in claim_groups else "false",
                "any_intermediate_claim": "true" if "intermediate" in claim_groups else "false",
            }
        )
    return output_rows


def prompt_for_batch(batch: list[dict[str, str]]) -> str:
    item_text = "\n".join(
        f"- {item['prompt_key']}: {item['definition']}" for item in ITEM_DEFINITIONS
    )
    word_records = [
        {
            "word": row["word"],
            "part_of_speech_hint": row["lexical_types"] or "unknown",
        }
        for row in batch
    ]
    return textwrap.dedent(
        f"""
        You are rating isolated English candidate words for a post hoc semantic stimulus audit.

        Important constraints:
        - Rate only the ordinary meaning of the word and the neutral part-of-speech hint.
        - Do not infer or use any experimental condition labels.
        - Use scores from 1.0 to 10.0.
        - Half-point values are allowed, for example 1.0, 1.5, 2.0, ..., 10.0.
        - Use the full scale when justified.
        - Return JSON only. Do not include markdown or explanations.

        Rating items:
        {item_text}

        Return exactly this JSON shape:
        {{
          "ratings": [
            {{
              "word": "example",
              "psychological_depth": 1.0,
              "self_reflective_relevance": 1.0,
              "existential_meaning": 1.0,
              "reflective_cognitive_process": 1.0,
              "personal_emotional_salience": 1.0,
              "concrete_observable": 10.0,
              "logistical_transactional": 10.0,
              "impersonal_neutral": 10.0
            }}
          ]
        }}

        Words to rate:
        {json.dumps(word_records, ensure_ascii=True, indent=2)}
        """
    ).strip()


def ollama_generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_ctx": 8192,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "")


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def half_point(value: Any) -> float:
    number = float(value)
    number = min(10.0, max(1.0, number))
    return round(number * 2.0) / 2.0


def validate_batch_response(batch: list[dict[str, str]], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    ratings = parsed.get("ratings")
    if not isinstance(ratings, list):
        raise ValueError("Response JSON does not contain a ratings list.")

    expected_words = {row["word"] for row in batch}
    by_word: dict[str, dict[str, Any]] = {}
    for row in ratings:
        if not isinstance(row, dict):
            continue
        word = str(row.get("word", "")).strip().lower()
        if word in expected_words:
            by_word[word] = row

    missing = sorted(expected_words - set(by_word))
    if missing:
        raise ValueError(f"Missing ratings for words: {', '.join(missing)}")

    normalized = []
    for word in sorted(expected_words):
        raw = by_word[word]
        out = {"word": word}
        for item in ITEM_DEFINITIONS:
            prompt_key = item["prompt_key"]
            if prompt_key not in raw:
                raise ValueError(f"Missing item {prompt_key} for word {word}")
            out[item["column"]] = half_point(raw[prompt_key])
        normalized.append(out)
    return normalized


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def model_show(model: str) -> str:
    try:
        completed = subprocess.run(
            ["ollama", "show", model],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return completed.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"ollama show failed: {exc}"


def main() -> int:
    global MODEL_PREFIX, RAW_SCORE_COLUMNS, RECODED_COLUMNS, COMPOSITE_COLUMN, ITEM_DEFINITIONS

    args = parse_args()
    MODEL_PREFIX = args.model_prefix.strip() or prefix_from_model_name(args.model)
    (
        RAW_SCORE_COLUMNS,
        RECODED_COLUMNS,
        COMPOSITE_COLUMN,
        ITEM_DEFINITIONS,
    ) = build_score_schema(MODEL_PREFIX)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = load_analyzed_word_metadata()
    randomized = metadata_rows[:]
    random.Random(args.seed).shuffle(randomized)
    batches = [
        randomized[start : start + args.batch_size]
        for start in range(0, len(randomized), args.batch_size)
    ]

    raw_jsonl_path = output_dir / f"{MODEL_PREFIX}_raw_responses.jsonl"
    existing = read_jsonl(raw_jsonl_path) if args.resume else []
    rated_by_word: dict[str, dict[str, Any]] = {}
    for existing_row in existing:
        if existing_row.get("ok") and isinstance(existing_row.get("normalized"), list):
            for rating in existing_row["normalized"]:
                rated_by_word[str(rating["word"]).lower()] = rating

    for batch_index, batch in enumerate(batches, start=1):
        batch_words = [row["word"] for row in batch]
        if all(word in rated_by_word for word in batch_words):
            print(f"Batch {batch_index}/{len(batches)} already rated; skipping.")
            continue

        prompt = prompt_for_batch(batch)
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                print(f"Rating batch {batch_index}/{len(batches)} with {len(batch)} words (attempt {attempt}).")
                response_text = ollama_generate(args.model, prompt)
                parsed = parse_json_response(response_text)
                normalized = validate_batch_response(batch, parsed)
                for rating in normalized:
                    rated_by_word[rating["word"]] = rating
                append_jsonl(
                    raw_jsonl_path,
                    {
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "model": args.model,
                        "batch_index": batch_index,
                        "words": batch_words,
                        "ok": True,
                        "response": parsed,
                        "normalized": normalized,
                    },
                )
                break
            except (urllib.error.URLError, json.JSONDecodeError, ValueError, TimeoutError) as exc:
                last_error = str(exc)
                append_jsonl(
                    raw_jsonl_path,
                    {
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "model": args.model,
                        "batch_index": batch_index,
                        "words": batch_words,
                        "ok": False,
                        "attempt": attempt,
                        "error": last_error,
                    },
                )
                time.sleep(2 * attempt)
        else:
            print(f"Failed batch {batch_index}: {last_error}", file=sys.stderr)
            return 1

    missing = sorted({row["word"] for row in metadata_rows} - set(rated_by_word))
    if missing:
        print(f"Missing ratings after run: {', '.join(missing)}", file=sys.stderr)
        return 1

    final_rows = []
    for row in metadata_rows:
        rating = dict(rated_by_word[row["word"]])
        combined = {**row, **rating}
        combined[f"{MODEL_PREFIX}_concrete_observable_depth_keyed"] = 11.0 - float(
            combined[f"{MODEL_PREFIX}_concrete_observable_reversed_raw"]
        )
        combined[f"{MODEL_PREFIX}_logistical_transactional_depth_keyed"] = 11.0 - float(
            combined[f"{MODEL_PREFIX}_logistical_transactional_reversed_raw"]
        )
        combined[f"{MODEL_PREFIX}_impersonal_neutral_depth_keyed"] = 11.0 - float(
            combined[f"{MODEL_PREFIX}_impersonal_neutral_reversed_raw"]
        )
        depth_keyed_columns = [
            f"{MODEL_PREFIX}_psychological_depth_raw",
            f"{MODEL_PREFIX}_self_reflective_relevance_raw",
            f"{MODEL_PREFIX}_existential_meaning_raw",
            f"{MODEL_PREFIX}_reflective_cognitive_process_raw",
            f"{MODEL_PREFIX}_personal_emotional_salience_raw",
            f"{MODEL_PREFIX}_concrete_observable_depth_keyed",
            f"{MODEL_PREFIX}_logistical_transactional_depth_keyed",
            f"{MODEL_PREFIX}_impersonal_neutral_depth_keyed",
        ]
        combined[COMPOSITE_COLUMN] = round(
            sum(float(combined[column]) for column in depth_keyed_columns) / len(depth_keyed_columns),
            3,
        )
        combined[f"{MODEL_PREFIX}_rating_model"] = args.model
        final_rows.append(combined)

    metadata_columns = [
        "word",
        "source_tables",
        "source_memberships",
        "roles",
        "lexical_types",
        "syntactic_frames",
        "simple_word_group",
        "simple_claim_group",
        "self_object_word_group",
        "self_object_claim_group",
        "cova_word_groups",
        "cova_claim_group",
        "cova_dimensions",
        "cova_polarities",
        "abstraction_word_groups",
        "abstraction_levels",
        "abstraction_min_level",
        "abstraction_max_level",
        "overall_claim_group",
        "any_depth_related_claim",
        "any_control_claim",
        "any_intermediate_claim",
    ]
    fieldnames = (
        metadata_columns
        + [f"{MODEL_PREFIX}_rating_model"]
        + RAW_SCORE_COLUMNS
        + RECODED_COLUMNS
        + [COMPOSITE_COLUMN]
    )
    dataset_path = output_dir / f"{MODEL_PREFIX}_word_depth_scores_wide.csv"
    write_csv(dataset_path, final_rows, fieldnames)

    codebook_path = output_dir / f"{MODEL_PREFIX}_word_depth_scores_codebook.json"
    codebook = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "model": args.model,
        "model_show": model_show(args.model),
        "n_words": len(final_rows),
        "source_files": {name: str(path.relative_to(PROJECT_DIR)) for name, path in SOURCE_FILES.items()},
        "cova_filter_for_analyzed_words": "lexical_type in ['verb', 'noun'] and word_group in ['depth', 'surface', 'neutral_control']",
        "score_range": "1.0 to 10.0, half-point increments allowed",
        "raw_score_columns": RAW_SCORE_COLUMNS,
        "reverse_keyed_raw_columns": [
            f"{MODEL_PREFIX}_concrete_observable_reversed_raw",
            f"{MODEL_PREFIX}_logistical_transactional_reversed_raw",
            f"{MODEL_PREFIX}_impersonal_neutral_reversed_raw",
        ],
        "recoding": "depth_keyed = 11 - reverse_keyed_raw",
        "composite_column": COMPOSITE_COLUMN,
        "composite_definition": "Mean of the five direct depth items and the three reverse-keyed items after recoding.",
        "item_definitions": ITEM_DEFINITIONS,
        "prompt_template": prompt_for_batch([{"word": "example", "lexical_types": "noun"}]),
    }
    codebook_path.write_text(json.dumps(codebook, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {dataset_path}")
    print(f"Wrote {codebook_path}")
    print(f"Wrote raw responses to {raw_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

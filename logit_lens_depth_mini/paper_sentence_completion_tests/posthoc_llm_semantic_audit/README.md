# Post hoc semantic stimulus audit

This audit asks whether the candidate words used in the sentence-completion study carry the intended semantic contrast. Six local language models rated 116 isolated words, giving 696 word-by-model observations. The analyses are reported in Supplementary Section S12 of the manuscript.

Start with the [HTML report](quarto/llm_semantic_audit_report.html), which contains the results, figures, analysis code, and recorded package versions. Download it and open it in a browser; it includes its resources and requires no model installation. [REPRODUCE.md](REPRODUCE.md) explains how to refit the statistical models, render the report, or collect new ratings.

## Design and interpretation

The rating prompt contained each word and neutral part-of-speech information, with no experimental group labels. Group membership was joined to the ratings afterwards. Each word received eight scores on a 1–10 scale, allowing half-points: psychological depth, self-reflective relevance, existential meaning, reflective cognitive process, personal or emotional salience, and three surface-level items concerning concreteness, logistics, and impersonality. The composite averages the five direct items and the three surface-level items after reverse-scoring them as `11 - score`.

Numeric item responses are clipped to the 1–10 range and rounded to the nearest half-point before scoring; composite scores are rounded to three decimal places. The raw response files preserve the model outputs before these steps.

Target words received higher composite ratings than comparison words across the model panel. The mixed models account for repeated ratings of words and variation between rating models. The item battery had high internal consistency, but the full eight-item multilevel CFA showed imperfect fit. A five-item model using only the direct indicators fit better. These results support the intended semantic direction of the word sets; they do not establish that the battery measures a single, fully validated psychological construct.

This is a post hoc stimulus audit based on model judgments. There was no human norming sample, and the ratings do not validate the sentence-completion task as a measure of human psychological depth.

## Results in the manuscript

| Supplementary table | Analysis | Saved results |
|---|---|---|
| S18 | Mixed-effects contrasts for the word sets | [Fixed effects and Kenward–Roger tests](outputs/statistics/all_local_llm_lme4_fixed_effects.csv) |
| S19 | Internal consistency | [Overall reliability](outputs/statistics/all_local_llm_depth_items_reliability_overall.csv); [reliability by model](outputs/statistics/all_local_llm_depth_items_reliability_by_model.csv) |
| S20 | Multilevel CFA | [Eight-item model fit](outputs/statistics/all_local_llm_depth_items_multilevel_cfa_fit_measures.csv); [five-item model fit](outputs/statistics/all_local_llm_depth_items_direct_only_multilevel_cfa_fit_measures.csv) |

Additional estimates, loadings, and model summaries are in [outputs/statistics](outputs/statistics/) and [reports](reports/).

## Files and models

- [data/source_word_lists](data/source_word_lists/) contains the four stimulus lists used to assemble the audit.
- [outputs/ratings](outputs/ratings/) contains the six model-specific datasets, combined datasets, raw response records, and codebooks.
- [scripts](scripts/) contains the rating script and the two R analysis scripts.
- [quarto](quarto/) contains the report source and the saved HTML.

The saved ratings use these model tags:

```text
gpt-oss:20b
phi4:latest
qwen3.5:9b-mlx
gemma4:e4b-mlx
mistral-small:22b-instruct-2409-q3_K_M
deepseek-r1:14b
```

Each `*_word_depth_scores_codebook.json` records the prompt template, item definitions, scoring rules, model description, and source filenames. The JSONL files retain response records and unsuccessful attempts, so the ratings can be checked against the original collection record.

The original codebooks refer to stimulus files as `data/<filename>.csv`. Those files are included here under `data/source_word_lists/<filename>.csv`, with the same filenames and contents. The codebooks retain their original metadata; the rating script reads the included `data/source_word_lists/` directory.

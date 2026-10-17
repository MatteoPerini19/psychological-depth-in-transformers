# Reproducing the semantic audit

The R commands use the locked library described in the repository's [environment guide](../../../ENVIRONMENT.md). Restore it from the repository root first and keep `R_LIBS_USER` set to that absolute library path.

The saved ratings and statistical outputs are sufficient to inspect the reported results. Model inference is needed only to collect new ratings.

From the repository root, enter this directory before running the commands below:

```bash
cd logit_lens_depth_mini/paper_sentence_completion_tests/posthoc_llm_semantic_audit
```

## Read the saved report

Open [quarto/llm_semantic_audit_report.html](quarto/llm_semantic_audit_report.html) in a browser. The file contains the report and its resources. It does not require R, Python, Quarto, or Ollama.

The [README](README.md) maps the CSV outputs to Supplementary Tables S18–S20. The combined long-format dataset is [all_local_llm_word_depth_scores_long.csv](outputs/ratings/all_local_llm_word_depth_scores_long.csv): one row per word and rating model, with 696 rows in total.

## Refit the statistical analyses

The saved report records R 4.5.2 and the following package versions:

| Purpose | Packages and recorded versions |
|---|---|
| Mixed models | `afex` 1.5.1, `emmeans` 2.0.3, `lme4` 2.0.1, `pbkrtest` 0.5.5 |
| Reliability and CFA | `psych` 2.6.1, `lavaan` 0.6.21 |
| Report | `dplyr` 1.2.0, `ggplot2` 4.0.2, `knitr` 1.51, `readr` 2.1.6, `scales` 1.4.0, `stringr` 1.6.0, `tidyr` 1.3.2 |

To install the current versions of the analysis packages, run this in R:

```r
install.packages(c("afex", "emmeans", "lme4", "pbkrtest", "psych", "lavaan"))
```

This installs the versions available from your configured repository; it does not recreate the recorded environment. Numerical results can vary with R and package versions. Use the saved CSVs as the reference outputs when comparing a new fit.

Run the mixed-effects analysis first, then the reliability and CFA analysis:

```bash
Rscript scripts/validate_all_local_llm_word_depth_audit_lme4.R
Rscript scripts/reliability_and_multilevel_cfa_local_llm_depth_items.R
```

The first script reads the six model-specific wide rating files and rebuilds the combined long and wide datasets. It fits the word-set contrasts, with word-level random intercepts and model-level random slopes. Fixed-effect tests use Type 3 conditional F-tests with the Kenward–Roger approximation; contrasts and confidence intervals use `emmeans` with Kenward–Roger degrees of freedom.

The second script reads the combined long dataset, computes alpha and omega, and fits the eight-item and five-item multilevel CFA models with word as the cluster. No language model is called by either R script.

These commands replace the derived files in `outputs/ratings/`, `outputs/statistics/`, and `reports/`. Work in a copy of the repository if you want to keep the included outputs beside your refits. Runtime depends on the R environment and hardware; no timing benchmark is supplied.

## Render the report from saved outputs

Rendering requires Quarto and these additional R packages:

```r
install.packages(c("dplyr", "ggplot2", "knitr", "readr", "scales", "stringr", "tidyr"))
```

Install the analysis packages listed above as well: the report reads their version metadata even when it does not refit models.

```bash
quarto render quarto/llm_semantic_audit_report.qmd
```

The report's `refit_models` parameter defaults to `false`, so this command reads the saved data and statistics. It replaces `quarto/llm_semantic_audit_report.html` with a standalone HTML file containing embedded resources. Keep the directory structure intact: the report resolves input paths relative to its own source directory.

## Collect new model ratings

Rating collection requires Python 3.11 or later, a local Ollama service at `http://localhost:11434`, and the requested models installed under the tags below. The Python script uses only the standard library. The `-mlx` tags refer to the models used in the original setup; availability and compatibility depend on the local model installation. Model weights and the Ollama runtime are not included here.

The following commands write a new set of ratings to a separate directory:

```bash
python3 scripts/run_local_llm_word_depth_audit.py --model gpt-oss:20b --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
python3 scripts/run_local_llm_word_depth_audit.py --model phi4:latest --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
python3 scripts/run_local_llm_word_depth_audit.py --model qwen3.5:9b-mlx --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
python3 scripts/run_local_llm_word_depth_audit.py --model gemma4:e4b-mlx --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
python3 scripts/run_local_llm_word_depth_audit.py --model mistral-small:22b-instruct-2409-q3_K_M --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
python3 scripts/run_local_llm_word_depth_audit.py --model deepseek-r1:14b --batch-size 12 --seed 20260705 --output-dir outputs/rerun/ratings --resume
```

Each command writes a wide rating CSV, a raw response JSONL file, and a codebook. `--resume` reuses successful responses in the selected output directory. If all words have already been rated there, it skips inference and rebuilds the CSV and codebook. Repeating a command after an interruption therefore resumes the same collection.

The script uses batches of 12 words, a word-order seed of `20260705`, temperature `0`, `top_p = 1`, and a context window of `8192`; it allows three attempts per batch. The seed controls word order, not model sampling. New ratings may differ because model tags, inference software, and hardware can change. The codebooks preserve model descriptions, but exact weight checksums and the Ollama version were not recorded. The included raw responses remain the record of the ratings used in the manuscript.

Once all six runs are complete, fit the new ratings without replacing the original statistical outputs:

```bash
Rscript scripts/validate_all_local_llm_word_depth_audit_lme4.R outputs/rerun/ratings outputs/rerun/statistics outputs/rerun/reports
Rscript scripts/reliability_and_multilevel_cfa_local_llm_depth_items.R outputs/rerun/ratings outputs/rerun/statistics outputs/rerun/reports
```

The supplied Quarto report reads the original `outputs/ratings/` and `outputs/statistics/` paths. Results from these optional runs can be inspected in `outputs/rerun/reports/` and `outputs/rerun/statistics/`.

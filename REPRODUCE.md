# Reproducing the analyses

The repository includes the saved model outputs, statistical results, and figures used in the manuscript. Start by checking those files. Downloading and running the language models is a separate, optional step.

The original analysis code and directory structure are preserved. Scripts write to their usual output folders, so use a separate copy of the repository if you want to compare a new run with the supplied results.

## Environment

From the repository root, create a Python environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLBACKEND=Agg
```

Use Python 3.11 or later, subject to the dependency versions in `requirements.txt`; the clean installation test used Python 3.11.0. Before the R steps, restore the locked R library and set `R_LIBS_USER` as described in [ENVIRONMENT.md](ENVIRONMENT.md). Keep that variable set in the same terminal while running the R commands.

## Check the saved results

These commands use the supplied CSV and Parquet files. They do not load models or download weights:

```bash
cd logit_lens_depth_mini/paper_sentence_completion_tests
python analysis/robustness/validate_reproduction.py
python scripts/03_build_paper_summary.py
```

The validation compares materials, layer conventions, and recomputed summaries with the saved results. Its expected result is 110 passing checks. A check of a separate release copy passed all 110 checks and regenerated the completion summary successfully. The validation writes `results/robustness/reproduction_validation_checks.csv` and `reports/reproduction_validation_report.md`. The second command writes the completion summary to `outputs/paper_test_confirmation_table.csv` and a Markdown companion.

Script 03 also writes `Paper Sentence Completion Test Package Summary.md` in the analysis folder. This legacy narrative is excluded from version control. Its “confirmatory” wording does not indicate preregistration, and its strength labels, also present in the summary CSV, are descriptive shortcuts. Use the manuscript and the reported robustness estimates to interpret the results.

To regenerate the three main figures, run this command **from the repository root**, with the R packages listed in `ENVIRONMENT.md` installed:

```bash
Rscript figures/main/create_main_figures.R
```

PNG and PDF files are written to `figures/main/`. Figure rendering can vary with the installed fonts and graphics libraries even when its numerical inputs are unchanged. In the clean test, the PDF content matched exactly; the PNG files showed small rasterization and font differences. See [the clean-installation record](release/clean_install_report.md).

The [analysis traceability table](release/analysis_traceability.csv) maps each manuscript table and figure to its inputs and scripts. Manuscript tables are formatted summaries of these files; the analysis scripts do not rebuild the Word document.

## Reanalyse the saved model outputs

Run the following commands from `logit_lens_depth_mini/paper_sentence_completion_tests`. They require no new model inference. Resampling is more computationally demanding than the checks above. Measured command times from the clean test are in [the installation report](release/clean_install_report.md).

```bash
python scripts/03_build_paper_summary.py
python scripts/04_run_posthoc_diagnostics.py
python analysis/robustness/build_analysis_dataset.py
python analysis/robustness/run_permutations.py
python analysis/robustness/run_hierarchical_models.py
python analysis/robustness/run_layer_window_sensitivity.py
python analysis/robustness/audit_manuscript_robustness.py
python analysis/robustness/validate_reproduction.py
```

These commands rebuild the dimension-level summaries, canonical datasets, permutation tests, bootstrap estimates, candidate sensitivity analyses, layer-window analyses, and template-family audit. The robustness scripts use settings and seeds in `analysis/robustness/config.yaml`. Script 04 has its own command-line defaults: `--seed 123` and `--n-shuffles 200`. Retain these defaults when comparing with the supplied results.

`candidate_logits.parquet` stores the canonical candidate-level data compactly. The loader prefers this file, so the larger CSV copy is not needed. Rebuilding the dataset also generates that CSV and records the current input locations in a `source_file` column. This column is provenance metadata and is not used in the statistical calculations.

The audit script retains some earlier table numbers in its output labels. Use `release/analysis_traceability.csv` for the current manuscript numbering.

## Rerun model inference

The completion analyses use GPT-2, DistilGPT-2, and Pythia 14M, 70M, 70M-deduped, and 160M. From `logit_lens_depth_mini/paper_sentence_completion_tests`, run:

```bash
python scripts/01_run_binary_completion_probes_all_models.py
python scripts/02_run_abstraction_gradient_all_models.py
python scripts/05_run_control_specificity_all_models.py
python analysis/robustness/run_tuned_lens.py
python analysis/robustness/run_patching_controls.py
```

The first three commands generate completion readouts, abstraction-control results, and lexical/context controls in `outputs/`. Then rerun the statistical commands above and the main-figure script to update summaries and figures.

The model scripts select CUDA, Apple MPS, or CPU according to availability. Several scripts also accept `--device` and `--models`; selecting a subset is useful for checking the environment but does not reproduce the complete six-model analysis. Weights may be downloaded on the first run. Model inference, particularly the intervention suite, is the most expensive part of this workflow; runtime depends on the hardware and is not benchmarked here.

The Tuned Lens script requires the base models and tokenizers to be cached locally; running the completion scripts first prepares that cache. It requests official lens checkpoints and records their status in `results/robustness/tuned_lens_status.csv`. The supplied calibration results cover GPT-2 and Pythia-70M-deduped.

After inference, inspect the console output, any `*_errors.csv` files in `outputs/` and `results/robustness/`, and the model coverage of the regenerated tables. A script can finish after only some models succeed. For Tuned Lens, check `tuned_lens_status.csv` before interpreting the result files: if no checkpoint loads, the script returns without replacing existing result CSVs. Those files still contain the supplied results, not estimates from the attempted rerun. Keep runs in separate copies so their status and outputs can be compared unambiguously.

**The manuscript's forward/reverse patching estimates use all 36 prompt pairs and are generated by `analysis/robustness/run_patching_controls.py`.** The suite imports model and intervention helpers from `scripts/06_run_self_object_activation_patching_all_models.py`. Running script 06 directly uses its own 12-pair default and does not regenerate the main patching estimates. Its supplied summary is retained as an input to the original validation checks.

## Semantic stimulus audit

The post hoc local-LLM audit has its own [reproduction instructions](logit_lens_depth_mini/paper_sentence_completion_tests/posthoc_llm_semantic_audit/REPRODUCE.md). Statistical reanalysis uses the supplied ratings; generating new ratings requires local Ollama models. These ratings describe model judgments of the candidate words and are not human validation.

## Comparing a new run with the supplied results

Fixed resampling seeds are recorded in the configuration. Numerical agreement also depends on package versions, model weights, tokenizers, and hardware. Model repository revisions were not pinned in the original scripts, and local LLM generation can vary between runs. Preserving the code therefore preserves the implemented analysis, but does not by itself guarantee bitwise-identical results in a new environment. Use the saved outputs and recorded environment as the reference when checking a rerun.

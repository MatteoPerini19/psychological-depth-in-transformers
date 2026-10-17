# Package contents

This package contains the code, materials, saved results, and figures supporting *Psychological Depth in Network Depth: Late-Intermediate Layers Support Self-Directed Inner-Life Sentence Completions*.

The analysis map in `analysis_traceability.csv` has one row for each of the two main tables, three main figures, and twenty supplementary tables. It identifies the relevant scripts, inputs, saved outputs, and rerun commands. The manuscript's current table numbering takes precedence over older table labels retained in the analysis code.

The original relative directory structure is preserved. The Python and R scripts are unchanged; the documentation explains how to use them in this package. The three canonical files below have portable `source_file` metadata in place of local absolute paths:

- `logit_lens_depth_mini/paper_sentence_completion_tests/results/robustness/analysis_dataset.csv`
- `logit_lens_depth_mini/paper_sentence_completion_tests/results/robustness/analysis_dataset.parquet`
- `logit_lens_depth_mini/paper_sentence_completion_tests/results/robustness/candidate_logits.parquet`

This metadata change does not affect the analysis values. Stimuli, model settings, seeds, raw rating responses, and saved numerical results retain their original contents.

## Inventory and checksums

`file_manifest.csv` lists each file's relative path, category, size in bytes, and SHA-256 hash. For files copied from the analysis project, it also records the source hash and identifies any documentation or provenance-metadata changes. The manifest excludes its own row to avoid a circular checksum.

`code_integrity.csv` provides the source-to-package comparison for the 20 Python and R scripts. `verification.md` describes the checks performed against the supplied results and the limits of those checks.

The root `README.md`, `REPRODUCE.md`, and `ENVIRONMENT.md` provide the reading order, commands, and dependency information. `CITATION.cff` provides citation metadata. Original code is covered by `LICENSE` (MIT), and original non-code content by `LICENSE-CC-BY-4.0.txt`, with scope and third-party terms described in `LICENSE_NOTES.md`.

`renv.lock` records the R package versions used for reproduction. The `environment/` folder records the complete Python and R package lists from the clean installation. `clean_install_report.md` describes that test and the remaining differences in PNG rendering.

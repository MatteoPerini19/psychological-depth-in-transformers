# Analysis environment

The saved Python environment record reports Python 3.11.0 on macOS arm64. The semantic-audit report records R 4.5.2. The versions below describe the original analysis environment. A clean installation was also tested on macOS arm64 with Python 3.11.0 and R 4.5.2; see `release/clean_install_report.md` for the commands and comparisons.

## Python

From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export MPLBACKEND=Agg
```

`requirements.txt` pins the packages used by the retained analyses. These requirements installed successfully from PyPI into an empty Python 3.11.0 environment, and `pip check` passed. `environment/python-packages-macos-arm64-py311.txt` records the complete installed package list, including transitive dependencies. Other operating systems have not been tested.

Set `MPLBACKEND=Agg` for command-line plotting. This writes figure files without opening a graphics window; the default interactive macOS backend aborted during the clean test.

The completion scripts use PyTorch and Hugging Face Transformers directly. They do not import the `transformer_lens` package. PyArrow is required for the supplied Parquet data; `tuned-lens` is required for the calibrated readout analysis. The local word-rating script uses the Python standard library and requires Python 3.11 or later.

A compact record of the original environment is retained at `logit_lens_depth_mini/paper_sentence_completion_tests/results/robustness/environment_info.json`.

## R and Quarto

The recorded R version is **4.5.2**. The analysis versions and the packages needed for figure and report rendering are listed below. The complete dependency versions are recorded in `renv.lock`:

| Package | Version |
| --- | --- |
| readr | 2.1.6 |
| dplyr | 1.2.0 |
| ggplot2 | 4.0.2 |
| afex | 1.5-1 |
| emmeans | 2.0.3 |
| lme4 | 2.0-1 |
| pbkrtest | 0.5.5 |
| psych | 2.6.1 |
| lavaan | 0.6-21 |
| knitr | 1.51 |
| scales | 1.4.0 |
| stringr | 1.6.0 |
| tidyr | 1.3.2 |
| rmarkdown | 2.30 |
| ragg | 1.5.0 |
| systemfonts | 1.3.1 |
| textshaping | 1.0.4 |

Restore the recorded versions with `renv.lock`. Archived packages may compile from source. On macOS, this requires Xcode Command Line Tools and the GNU Fortran compiler matching the R build; see the official [R for Mac tools page](https://mac.r-project.org/tools/). The clean test used GNU Fortran 14.2 for R 4.5.2. For arm64 R, use a compiler that can build for arm64. The graphics stack also requires FreeType, HarfBuzz, FriBidi, PNG, JPEG, and TIFF development libraries.

For source builds using Homebrew libraries on Apple Silicon, the clean test also needed the library search paths below. This project-local file leaves the system Makevars unchanged. Configure it before restoring packages:

```sh
mkdir -p .R-library
cat > .R-library/Makevars <<'MAKE'
LDFLAGS = -L/opt/homebrew/lib -L/opt/R/arm64/lib
MAKE
export R_MAKEVARS_USER="$PWD/.R-library/Makevars"
```

Adapt or omit that block when using another native-library prefix. From the repository root, restore the R packages:

```sh
mkdir -p .R-library
export R_LIBS_USER="$PWD/.R-library"
Rscript - <<'RS'
install.packages("renv", repos="https://cloud.r-project.org")
lock <- renv::lockfile_read("renv.lock")
renv::restore(
  lockfile="renv.lock", library=Sys.getenv("R_LIBS_USER"),
  packages=names(lock$Packages), rebuild=TRUE, prompt=FALSE
)
RS
```

The lockfile records 116 dependency versions for the figures, statistical analyses, and Quarto report, including `rmarkdown` and the original `ragg` PNG renderer. The clean restore used `renv` 1.3.1.

Keep `R_LIBS_USER` set while running the R commands. In a new terminal, repeat the `export` command from the repository root; its absolute path continues to work after changing into an analysis subfolder. The `.R-library/` directory is local and excluded from version control.

Installing the latest CRAN packages without the lockfile changed the CFA estimates and figure rendering in the clean test. Use the locked versions when comparing a rerun with the supplied results.

Quarto 1.7.31 was available during preparation. It is needed only to render the `.qmd` report; the saved HTML can be opened directly in a browser.

## Model inference

The six completion models are GPT-2, DistilGPT-2, and Pythia 14M, 70M, 70M-deduped, and 160M. Model weights are downloaded or read from local caches by the existing scripts. The Tuned Lens script expects its base models in the local Hugging Face cache. No weights or model caches are distributed here.

The word-rating audit additionally requires Ollama and the six model tags recorded in its codebooks. Checkpoint revisions and Ollama weight digests were not frozen in the original run. Random seeds in the analysis scripts should be distinguished from an inference seed: the word-rating script fixes word order but does not send an inference seed to Ollama.

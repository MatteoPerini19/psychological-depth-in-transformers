# Clean installation and reproduction

Tested on 17 October 2026 with macOS arm64, Python 3.11.0, R 4.5.2, and Quarto 1.7.31. All checks used separate copies of the package. The 20 analysis scripts, materials, and supplied results were preserved.

## Environment

The Python environment started with only pip and setuptools. The unmodified requirements.txt installed from public PyPI without a package cache. Dependency checks and imports of the analysis libraries passed. The complete installed package list is recorded in environment/python-packages-macos-arm64-py311.txt.

The R package libraries started empty, with only R's base and recommended packages visible. Installing current CRAN packages changed the CFA estimates and figure rendering. Restoring the original versions resolved the numerical differences. The renv.lock file records 116 dependency versions, including report dependencies and the original ragg PNG renderer. All nonstandard R packages used in the locked run were installed in the new library; environment/r-packages-macos-arm64.csv records their versions.

Archived packages required source compilation. The test used GNU Fortran 14.2 in a temporary extraction, the available native graphics libraries, and a local Makevars configuration. No compiler was installed into the system. The [environment guide](../ENVIRONMENT.md) documents these prerequisites and the locked restore command.

Python plotting used MPLBACKEND=Agg. The default interactive macOS backend aborted during diagnostics plotting. Both original plotting functions were checked with Agg, followed by a successful complete rerun of the unchanged script with its original defaults.

## Workflows and agreement

The Python saved-output workflow in [REPRODUCE.md](../REPRODUCE.md) completed, including the full diagnostics script, dataset rebuild, permutation tests, hierarchical bootstrap analyses, layer-window checks, manuscript audit, and final validation. The figures, both semantic-audit R scripts, and the Quarto report were also regenerated.

| Output | Comparison |
| --- | --- |
| Validator | 110/110 checks passed before and after reanalysis |
| Python data and robustness results | All analysis values in 25 regenerated included files match exactly; 22 files are byte-identical and three differ only in provenance paths |
| Taxonomy dimension results | The additional 54-row CSV is byte-identical |
| Semantic-audit results | All 13 regenerated CSV files are byte-identical |
| Main PDF figures | Content streams and text match exactly; only creation/modification timestamps differ |
| Main PNG figures | Same numerical inputs and figure content; small font and rasterization differences remain |
| HTML report | 39 chunks completed; all ten report tables have identical normalized text |

The PNGs are not byte-identical after rebuilding the graphics packages against the available native libraries. Changed pixels account for 0.50%–1.50% of each image; all three figures were also inspected visually. The unchanged PDF content separately checks plotted values and vector geometry. The supplied publication figures remain the reference files.

The original singular-fit and CFA diagnostics were reproduced. Numerical agreement does not resolve the associated modeling limitations.

## Measured runtimes

These are observations from this machine. Some checks ran concurrently.

| Step | Time |
| --- | --- |
| Python installation | 72.6 seconds |
| Main Python checks and robustness commands | 5.1 minutes |
| Full diagnostics script, original 200-shuffle default | 13.5 minutes |
| Locked R restore | 3.6 minutes, plus 1.4 minutes for graphics packages |
| Main figures | 2.2 seconds |
| Both semantic-audit R scripts | 29.8 seconds |
| Quarto rendering | 5.0 seconds |

## Scope

This test covers installation and analysis of the supplied model outputs and ratings. Transformer inference, activation-patching inference, and Ollama generation were not rerun. No model checkpoints were downloaded. Other operating systems and architectures were not tested.

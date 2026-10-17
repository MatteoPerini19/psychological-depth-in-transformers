# Verification record

The package was checked in separate copies using the original analysis environment on 16 October 2026 and a clean installation on 17 October 2026. The installed package's code, data, and figures were preserved throughout these checks.

| Check | Result |
| --- | --- |
| Analysis code | All 20 Python and R scripts match the original analysis sources byte for byte |
| Saved-result validation | 110 of 110 checks passed before and after the clean Python reanalysis |
| Python reanalysis | All analysis values in 25 regenerated included data/result files match exactly; three files differ only in their `source_file` provenance paths |
| Taxonomy dimension reanalysis | The 54-row CSV matches byte for byte after the complete diagnostics command |
| Semantic-audit reanalysis | All 13 regenerated CSV files match the supplied files byte for byte with the locked R versions |
| Supplied figures | The three supplied PNGs match the images embedded in the manuscript |
| Clean figure generation | All three PDFs have identical content streams and text; only timestamps differ. The PNGs have small font and rasterization differences |
| Canonical data | All nonprovenance values, row order, and schemas are preserved across 9,720 aggregate rows and 234,972 candidate rows |
| Manuscript traceability | All 22 tables and three figures have supporting files; 963 numeric or categorical cells were checked against the supplied data |
| Raw semantic responses | All 696 word/model observations trace to the raw responses through the original scoring rules; 11,832 scalar comparisons passed |
| HTML report | Quarto completed all 39 chunks with the clean R library; all ten report tables match the supplied HTML table text |
| Public package | No private machine paths, credentials, or excluded cache/environment files found, including decoded Parquet contents and image/PDF metadata |

The mixed-effects and CFA reanalyses reproduce the existing statistical diagnostics, including singular fits, a negative variance estimate, and unavailable robust fit indices. Numerical reproduction does not resolve these modeling limitations.

The [clean-installation report](clean_install_report.md) gives the dependency setup, commands, timings, and comparison results. Transformer and Ollama inference were not rerun; checkpoint revisions were not frozen in the original scripts. Other operating systems and CPU architectures were not tested.

`code_integrity.csv` records the source and package hashes for every included Python and R script. `file_manifest.csv` records every package file except the manifest itself. No original analysis source was changed while preparing or checking this package.

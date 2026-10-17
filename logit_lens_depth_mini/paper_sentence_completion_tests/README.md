# Sentence-completion analyses

This folder contains the model readouts, interventions, and statistical analyses for the psychological-depth manuscript. The primary contrast compares inner-life continuations after self-directed and object-directed sentence stems. A depth-taxonomy probe tests convergence, and an abstraction-gradient probe tests whether a generic shift toward abstract words explains the profile.

The six models are GPT-2, DistilGPT-2, Pythia-14M, Pythia-70M, Pythia-70M-deduped, and Pythia-160M. Materials include 72 self/object stems arranged in 36 pairs drawn from six template-pair families.

## Reading the results

Candidate probabilities are normalized over the specified target and control words, not the full model vocabulary. The principal quantity is a logsumexp margin between the two candidate sets. For the primary contrast, the late-minus-middle change after object-directed stems is subtracted from the corresponding change after self-directed stems.

Layer 0 denotes the embedding state; layers 1 through N denote transformer block outputs. The middle window is blocks 3–6 for 12-layer models and blocks 2–3 for six-layer models. The late-intermediate window is blocks 8–11 and 4–5, respectively. The final block is summarized separately. Alternative windows are evaluated in the sensitivity analyses.

The five-model positive pattern has a Pythia-14M reversal. Template-family and candidate-label audits further qualify generalization beyond the sampled wording. Readout accessibility, causal effects of interventions, and psychological interpretation are distinct claims.

## Files and manuscript links

| Location | Contents | Manuscript |
| --- | --- | --- |
| `data/` | Sentence stems and candidate word lists | Methods |
| `outputs/cross_model_sentence_completion_*.csv` | Candidate logits, layer-level results, summaries, and token checks | Main Table 1; Figures 1–2; supplementary Tables S1 and S16 |
| `outputs/abstraction_gradient_cross_model_*.csv` | Abstraction-gradient control | Main Table 2 and specificity results |
| `outputs/posthoc_cova_dimension_robustness.csv` | Depth-taxonomy dimension estimates | Supplementary Table S2 |
| `outputs/control_*.csv` | Lexical matching and intentional/mechanical controls | Supplementary Tables S15 and S17 |
| `results/robustness/` | Calibration, resampling, permutations, window sensitivity, and patching | Main Table 2; Figure 3; supplementary Tables S1 and S3–S14 |
| `posthoc_llm_semantic_audit/` | Blinded local-model ratings and their statistical analysis | Supplementary Section S12, Tables S18–S20 |

The [traceability table](../../release/analysis_traceability.csv) gives exact file-to-result links. The three main figures and their R script are in [`figures/main/`](../../figures/main/).

## Running the analyses

Follow the repository's [reproduction guide](../../REPRODUCE.md) for environment setup, saved-output checks, statistical reanalysis, and optional model inference.

The numbered scripts provide the completion probes, summary construction, dimension summaries, and specificity controls. `analysis/robustness/` contains the calibration, resampling, and intervention analyses, with settings in `config.yaml`.

For the main causal result, use `analysis/robustness/run_patching_controls.py`: it runs forward, reverse, and control interventions across all 36 pairs. Script 06 supplies imported helpers; its direct-run default uses 12 pairs. The main estimates are in `results/robustness/patching_bootstrap.csv`.

The code retains its original relative paths and writes results back into this folder. Keep the directory structure intact. The saved files allow the reported analyses to be inspected without downloading model weights.

# Main Figure Captions

## Figure 1

*Layerwise Trajectory of the Primary Self-Directed Inner-Life Contrast*

Note. Layerwise trajectory of the primary self-directed inner-life contrast. Late-intermediate layers preferentially supported inner-life continuations in five of six models, with Pythia-14M providing the boundary case. Lines show mean candidate-restricted logit margins for self-directed and object-directed stems at each normalized depth. Positive values indicate greater relative support for inner-life over object/logistical candidates within the predefined candidate pool. Gray shading marks the middle-layer window and blue shading marks the late-intermediate window used for the late-minus-middle contrast. Source data: `logit_lens_depth_mini/paper_sentence_completion_tests/outputs/cross_model_sentence_completion_summary_by_layer.csv`.

Files:
- `figures/main/figure_1_layerwise_self_object.png`
- `figures/main/figure_1_layerwise_self_object.pdf`

## Figure 2

*Late-Minus-Middle Effects for the Primary and Cova-Derived Probes*

Note. Late-minus-middle effects for the primary self-directed inner-life probe and the Cova-derived depth-positive probe. The convergence of these two probes supports the construct-specific depth interpretation. Points show model-level late-minus-middle logit-margin effects. Error bars show 95% confidence intervals computed from the saved stem/unit-level standard deviations in `cross_model_sentence_completion_summary_by_model.csv`; they are not model-level bootstrap intervals. Positive values indicate larger late-intermediate than middle-layer support for the relevant contrast.

Files:
- `figures/main/figure_2_primary_and_cova_effects.png`
- `figures/main/figure_2_primary_and_cova_effects.pdf`

## Figure 3

*Forward and Reverse Residual-Stream Patching Effects*

Note. Residual-stream patching shows causal contribution to the measured margin. Self-directed residual states increased the inner-life/object margin when patched into object-directed computations, whereas reverse patches moved the margin in the opposite direction. Points show the mean late-intermediate raw delta in the inner-life/object logit margin for residual-stream post-block patching. Error bars show 95% bootstrap intervals from `patching_bootstrap.csv`.

Files:
- `figures/main/figure_3_patching_forward_reverse.png`
- `figures/main/figure_3_patching_forward_reverse.pdf`

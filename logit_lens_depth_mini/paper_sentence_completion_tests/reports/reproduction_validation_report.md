# Reproduction Validation Report

Checks run: 110.
Checks passed: 110.
Checks failed: 0.

The canonical robustness dataset preserves the existing layer convention: embedding is `layer_index = 0`, transformer block outputs are `1..N`, and the final block is summarized separately from late-intermediate layers.

All reproduction checks passed within numerical tolerance.
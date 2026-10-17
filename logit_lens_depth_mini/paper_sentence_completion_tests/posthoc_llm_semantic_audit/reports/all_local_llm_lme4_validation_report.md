# All-local-LLM post hoc semantic stimulus audit

- Models: deepseek-r1:14b, gemma4:e4b-mlx, gpt-oss:20b, mistral-small:22b-instruct-2409-q3_K_M, phi4:latest, qwen3.5:9b-mlx
- Long observations: 696
- Unique words: 116
- Model formula, binary contrasts: `psychological_depth_composite ~ depth_related_factor + (1 + depth_related_factor || rating_model) + (1 | word)`
- Model formula, abstraction gradient: `psychological_depth_composite ~ abstraction_max_level + (1 + abstraction_max_level || rating_model) + (1 | word)`
- Fixed-effect p values use Type 3 conditional F-tests with Kenward-Roger approximation via `afex`/`pbkrtest`.
- Contrasts, slopes, and confidence intervals use `emmeans` with Kenward-Roger df.

## Fixed effects

- reflective_vs_practical_words: estimate = 4.548, SE = 0.347, 95% CI [3.818, 5.278], F(1, 17.48) = 172.085, p < .001, observations = 120, words = 20, models = 6, singular = FALSE
- inner_life_vs_object_words: estimate = 5.853, SE = 0.379, 95% CI [5.051, 6.655], F(1, 16.34) = 238.581, p < .001, observations = 120, words = 20, models = 6, singular = FALSE
- cova_depth_vs_control_words: estimate = 4.420, SE = 0.467, 95% CI [3.485, 5.354], F(1, 59.68) = 89.553, p < .001, observations = 372, words = 62, models = 6, singular = TRUE
- clean_overall_depth_vs_control_words: estimate = 4.551, SE = 0.274, 95% CI [4.007, 5.096], F(1, 96.63) = 275.162, p < .001, observations = 606, words = 101, models = 6, singular = TRUE
- abstraction_gradient_ordering: estimate = 1.543, SE = 0.151, 95% CI [1.242, 1.844], F(1, 72.39) = 104.511, p < .001, observations = 450, words = 75, models = 6, singular = TRUE

## Interpretation boundary

These are post hoc LLM-based semantic stimulus-audit results. They assess whether local LLMs rate the analyzed target words as semantically deeper than control words. They should not be described as human construct validation.


# Reliability and multilevel CFA for local-LLM depth items

- Rows: 696
- Words/clusters: 116
- Rating models: 6
- Indicators: y_psychological_depth, y_self_reflective_relevance, y_existential_meaning, y_reflective_cognitive_process, y_personal_emotional_salience, y_concrete_observable_depth_keyed, y_logistical_transactional_depth_keyed, y_impersonal_neutral_depth_keyed

## Reliability

- All word-by-model rows: alpha = 0.972, omega total = 0.973, omega hierarchical = 0.973
- Word-mean scores: alpha = 0.985, omega total = 0.986, omega hierarchical = 0.986

## Multilevel CFA

Model:

```r
level: 1
  within_depth =~ y_psychological_depth + y_self_reflective_relevance + y_existential_meaning + y_reflective_cognitive_process + y_personal_emotional_salience + y_concrete_observable_depth_keyed + y_logistical_transactional_depth_keyed + y_impersonal_neutral_depth_keyed

level: 2
  between_depth =~ y_psychological_depth + y_self_reflective_relevance + y_existential_meaning + y_reflective_cognitive_process + y_personal_emotional_salience + y_concrete_observable_depth_keyed + y_logistical_transactional_depth_keyed + y_impersonal_neutral_depth_keyed

fit_ml <- lavaan::cfa(
  model = ml_cfa_model,
  data = dat,
  cluster = "word",
  estimator = "MLR",
  std.lv = TRUE,
  missing = "fiml"
)
```

- Fit: CFI = 0.845, TLI = 0.783, RMSEA = 0.159, SRMR = 0.158, SRMR within = 0.129, SRMR between = 0.029
- Direct-item-only sensitivity: CFI = 0.951, TLI = 0.902, RMSEA = 0.143, SRMR within = 0.019, SRMR between = 0.039

## Interpretation boundary

These analyses assess internal consistency and single-factor measurement structure of an LLM-based semantic audit battery. They support measurement coherence of the audit scores, but they do not prove construct validity in the strong human-validation sense.


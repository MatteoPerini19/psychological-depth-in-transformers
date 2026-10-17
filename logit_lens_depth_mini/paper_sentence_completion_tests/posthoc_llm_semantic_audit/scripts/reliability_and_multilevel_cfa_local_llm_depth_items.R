#!/usr/bin/env Rscript

# Reliability and multilevel CFA for the local-LLM semantic depth item battery.
#
# Rows are word-by-model ratings: 116 words x 6 local LLMs = 696 observations.
# The primary multilevel CFA clusters by word, because each word is rated
# repeatedly by multiple models. This estimates a within-word factor (model
# deviations around each word) and a between-word factor (word-level semantic
# differences).

args <- commandArgs(trailingOnly = TRUE)
all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- all_args[grepl("^--file=", all_args)]
if (length(file_arg) == 0) stop("Could not determine script path from commandArgs().")
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
project_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)

ratings_dir <- if (length(args) >= 1) {
  args[[1]]
} else {
  file.path(project_dir, "outputs", "ratings")
}
ratings_dir <- normalizePath(ratings_dir, mustWork = TRUE)
output_dir <- if (length(args) >= 2) args[[2]] else file.path(project_dir, "outputs", "statistics")
report_dir <- if (length(args) >= 3) args[[3]] else file.path(project_dir, "reports")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
input_path <- file.path(ratings_dir, "all_local_llm_word_depth_scores_long.csv")

if (!requireNamespace("psych", quietly = TRUE)) stop("Package 'psych' is required.")
if (!requireNamespace("lavaan", quietly = TRUE)) stop("Package 'lavaan' is required.")

dat_raw <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)

indicator_map <- c(
  y_psychological_depth = "psychological_depth_raw",
  y_self_reflective_relevance = "self_reflective_relevance_raw",
  y_existential_meaning = "existential_meaning_raw",
  y_reflective_cognitive_process = "reflective_cognitive_process_raw",
  y_personal_emotional_salience = "personal_emotional_salience_raw",
  y_concrete_observable_depth_keyed = "concrete_observable_depth_keyed",
  y_logistical_transactional_depth_keyed = "logistical_transactional_depth_keyed",
  y_impersonal_neutral_depth_keyed = "impersonal_neutral_depth_keyed"
)

missing_cols <- setdiff(c("word", "rating_model", unname(indicator_map)), names(dat_raw))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

dat <- data.frame(
  word = factor(dat_raw$word),
  rating_model = factor(dat_raw$rating_model),
  stringsAsFactors = FALSE
)
for (target in names(indicator_map)) {
  dat[[target]] <- suppressWarnings(as.numeric(dat_raw[[indicator_map[[target]]]]))
}
indicator_names <- names(indicator_map)
dat <- dat[complete.cases(dat[, indicator_names]), , drop = FALSE]

safe_alpha <- function(items) {
  out <- psych::alpha(items, check.keys = FALSE, warnings = FALSE)
  unname(out$total$raw_alpha)
}

safe_omega <- function(items) {
  out <- tryCatch(
    psych::omega(items, nfactors = 1, fm = "minres", plot = FALSE, warnings = FALSE),
    error = function(e) e
  )
  if (inherits(out, "error")) {
    return(c(omega_total = NA_real_, omega_hierarchical = NA_real_))
  }
  omega_total <- if (!is.null(out$omega.tot)) out$omega.tot else NA_real_
  omega_h <- if (!is.null(out$omega_h)) out$omega_h else NA_real_
  c(omega_total = unname(omega_total), omega_hierarchical = unname(omega_h))
}

reliability_row <- function(label, items, n_words = NA_integer_, n_models = NA_integer_) {
  alpha <- safe_alpha(items)
  omega <- safe_omega(items)
  data.frame(
    analysis = label,
    n_observations = nrow(items),
    n_words = n_words,
    n_models = n_models,
    cronbach_alpha = alpha,
    mcdonald_omega_total = omega[["omega_total"]],
    mcdonald_omega_hierarchical = omega[["omega_hierarchical"]],
    stringsAsFactors = FALSE
  )
}

items_all <- dat[, indicator_names, drop = FALSE]
word_means <- aggregate(dat[, indicator_names, drop = FALSE], by = list(word = dat$word), FUN = mean)
items_word_mean <- word_means[, indicator_names, drop = FALSE]

reliability_overall <- rbind(
  reliability_row(
    "all_word_by_model_rows",
    items_all,
    n_words = length(unique(dat$word)),
    n_models = length(unique(dat$rating_model))
  ),
  reliability_row(
    "word_mean_scores",
    items_word_mean,
    n_words = nrow(word_means),
    n_models = length(unique(dat$rating_model))
  )
)

per_model_reliability <- do.call(
  rbind,
  lapply(split(dat, dat$rating_model), function(model_dat) {
    out <- reliability_row(
      paste0("model_", unique(as.character(model_dat$rating_model))),
      model_dat[, indicator_names, drop = FALSE],
      n_words = length(unique(model_dat$word)),
      n_models = 1
    )
    out$rating_model <- unique(as.character(model_dat$rating_model))
    out
  })
)
per_model_reliability <- per_model_reliability[
  ,
  c("rating_model", setdiff(names(per_model_reliability), "rating_model")),
  drop = FALSE
]

one_factor_items <- paste(indicator_names, collapse = " + ")
ml_cfa_model <- paste0(
  "level: 1\n",
  "  within_depth =~ ", one_factor_items, "\n\n",
  "level: 2\n",
  "  between_depth =~ ", one_factor_items, "\n"
)

fit_multilevel_cfa <- function(model_syntax) {
  tryCatch(
    lavaan::cfa(
      model = model_syntax,
      data = dat,
      cluster = "word",
      estimator = "MLR",
      std.lv = TRUE,
      missing = "fiml"
    ),
    error = function(e) {
      message("MLR multilevel CFA failed; retrying with ML. Error: ", e$message)
      lavaan::cfa(
        model = model_syntax,
        data = dat,
        cluster = "word",
        estimator = "ML",
        std.lv = TRUE,
        missing = "fiml"
      )
    }
  )
}

fit_ml <- fit_multilevel_cfa(ml_cfa_model)

direct_indicator_names <- indicator_names[1:5]
direct_factor_items <- paste(direct_indicator_names, collapse = " + ")
direct_ml_cfa_model <- paste0(
  "level: 1\n",
  "  within_depth =~ ", direct_factor_items, "\n\n",
  "level: 2\n",
  "  between_depth =~ ", direct_factor_items, "\n"
)
fit_direct_ml <- fit_multilevel_cfa(direct_ml_cfa_model)

fit_names <- c(
  "chisq", "df", "pvalue",
  "cfi", "tli", "rmsea", "srmr",
  "cfi.robust", "tli.robust", "rmsea.robust",
  "srmr_within", "srmr_between",
  "aic", "bic"
)
available_fit_names <- fit_names[fit_names %in% names(lavaan::fitMeasures(fit_ml))]
fit_measures <- data.frame(
  measure = available_fit_names,
  value = as.numeric(lavaan::fitMeasures(fit_ml, available_fit_names)),
  row.names = NULL
)

loadings <- lavaan::standardizedSolution(fit_ml)
loadings <- loadings[loadings$op == "=~", , drop = FALSE]

direct_fit_measures <- data.frame(
  measure = available_fit_names,
  value = as.numeric(lavaan::fitMeasures(fit_direct_ml, available_fit_names)),
  row.names = NULL
)

direct_loadings <- lavaan::standardizedSolution(fit_direct_ml)
direct_loadings <- direct_loadings[direct_loadings$op == "=~", , drop = FALSE]

icc <- tryCatch(
  {
    icc_value <- lavaan::lavInspect(fit_ml, "icc")
    data.frame(indicator = names(icc_value), icc = as.numeric(icc_value), row.names = NULL)
  },
  error = function(e) data.frame(indicator = character(), icc = numeric())
)

summary_path <- file.path(report_dir, "all_local_llm_depth_items_multilevel_cfa_summary.txt")
sink(summary_path)
cat("LOCAL LLM DEPTH ITEMS: RELIABILITY AND MULTILEVEL CFA\n")
cat("======================================================\n\n")
cat("Input: outputs/ratings/all_local_llm_word_depth_scores_long.csv\n")
cat("Rows after complete-case filtering: ", nrow(dat), "\n", sep = "")
cat("Words: ", length(unique(dat$word)), "\n", sep = "")
cat("Rating models: ", paste(levels(dat$rating_model), collapse = ", "), "\n\n", sep = "")
cat("Indicators:\n")
print(indicator_names)
cat("\nReliability overall:\n")
print(reliability_overall)
cat("\nReliability by model:\n")
print(per_model_reliability)
cat("\nMultilevel CFA model syntax:\n")
cat(ml_cfa_model, "\n")
cat("\nMultilevel CFA summary:\n")
print(summary(fit_ml, fit.measures = TRUE, standardized = TRUE))
cat("\nDirect-item-only multilevel CFA model syntax:\n")
cat(direct_ml_cfa_model, "\n")
cat("\nDirect-item-only multilevel CFA summary:\n")
print(summary(fit_direct_ml, fit.measures = TRUE, standardized = TRUE))
sink()

write.csv(
  reliability_overall,
  file.path(output_dir, "all_local_llm_depth_items_reliability_overall.csv"),
  row.names = FALSE
)
write.csv(
  per_model_reliability,
  file.path(output_dir, "all_local_llm_depth_items_reliability_by_model.csv"),
  row.names = FALSE
)
write.csv(
  fit_measures,
  file.path(output_dir, "all_local_llm_depth_items_multilevel_cfa_fit_measures.csv"),
  row.names = FALSE
)
write.csv(
  loadings,
  file.path(output_dir, "all_local_llm_depth_items_multilevel_cfa_standardized_loadings.csv"),
  row.names = FALSE
)
write.csv(
  direct_fit_measures,
  file.path(output_dir, "all_local_llm_depth_items_direct_only_multilevel_cfa_fit_measures.csv"),
  row.names = FALSE
)
write.csv(
  direct_loadings,
  file.path(output_dir, "all_local_llm_depth_items_direct_only_multilevel_cfa_standardized_loadings.csv"),
  row.names = FALSE
)
write.csv(
  icc,
  file.path(output_dir, "all_local_llm_depth_items_multilevel_cfa_icc.csv"),
  row.names = FALSE
)

fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
fit_lookup <- setNames(fit_measures$value, fit_measures$measure)
get_fit <- function(name) if (name %in% names(fit_lookup)) fit_lookup[[name]] else NA_real_

report <- c(
  "# Reliability and multilevel CFA for local-LLM depth items",
  "",
  paste0("- Rows: ", nrow(dat)),
  paste0("- Words/clusters: ", length(unique(dat$word))),
  paste0("- Rating models: ", length(unique(dat$rating_model))),
  paste0("- Indicators: ", paste(indicator_names, collapse = ", ")),
  "",
  "## Reliability",
  "",
  paste0(
    "- All word-by-model rows: alpha = ", fmt(reliability_overall$cronbach_alpha[reliability_overall$analysis == "all_word_by_model_rows"]),
    ", omega total = ", fmt(reliability_overall$mcdonald_omega_total[reliability_overall$analysis == "all_word_by_model_rows"]),
    ", omega hierarchical = ", fmt(reliability_overall$mcdonald_omega_hierarchical[reliability_overall$analysis == "all_word_by_model_rows"])
  ),
  paste0(
    "- Word-mean scores: alpha = ", fmt(reliability_overall$cronbach_alpha[reliability_overall$analysis == "word_mean_scores"]),
    ", omega total = ", fmt(reliability_overall$mcdonald_omega_total[reliability_overall$analysis == "word_mean_scores"]),
    ", omega hierarchical = ", fmt(reliability_overall$mcdonald_omega_hierarchical[reliability_overall$analysis == "word_mean_scores"])
  ),
  "",
  "## Multilevel CFA",
  "",
  "Model:",
  "",
  "```r",
  ml_cfa_model,
  "fit_ml <- lavaan::cfa(",
  "  model = ml_cfa_model,",
  "  data = dat,",
  "  cluster = \"word\",",
  "  estimator = \"MLR\",",
  "  std.lv = TRUE,",
  "  missing = \"fiml\"",
  ")",
  "```",
  "",
  paste0(
    "- Fit: CFI = ", fmt(get_fit("cfi")),
    ", TLI = ", fmt(get_fit("tli")),
    ", RMSEA = ", fmt(get_fit("rmsea")),
    ", SRMR = ", fmt(get_fit("srmr")),
    ", SRMR within = ", fmt(get_fit("srmr_within")),
    ", SRMR between = ", fmt(get_fit("srmr_between"))
  ),
  paste0(
    "- Direct-item-only sensitivity: CFI = ",
    fmt(direct_fit_measures$value[direct_fit_measures$measure == "cfi"]),
    ", TLI = ",
    fmt(direct_fit_measures$value[direct_fit_measures$measure == "tli"]),
    ", RMSEA = ",
    fmt(direct_fit_measures$value[direct_fit_measures$measure == "rmsea"]),
    ", SRMR within = ",
    fmt(direct_fit_measures$value[direct_fit_measures$measure == "srmr_within"]),
    ", SRMR between = ",
    fmt(direct_fit_measures$value[direct_fit_measures$measure == "srmr_between"])
  ),
  "",
  "## Interpretation boundary",
  "",
  "These analyses assess internal consistency and single-factor measurement structure of an LLM-based semantic audit battery. They support measurement coherence of the audit scores, but they do not prove construct validity in the strong human-validation sense.",
  ""
)
writeLines(report, file.path(report_dir, "all_local_llm_depth_items_reliability_cfa_report.md"))

cat("Wrote reliability and multilevel CFA outputs to ", output_dir, "\n", sep = "")
cat("Summary file: ", summary_path, "\n", sep = "")

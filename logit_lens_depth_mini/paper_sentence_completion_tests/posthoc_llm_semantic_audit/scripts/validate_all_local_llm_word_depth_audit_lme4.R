#!/usr/bin/env Rscript

# Multi-model post hoc semantic audit with local Ollama LLMs.
# The main tests use mixed models with random intercepts and random slopes
# by rating model, plus a word-level random intercept for repeated ratings
# of the same word across models. Fixed-effect tests use Type 3 conditional
# F-tests with Kenward-Roger degrees of freedom via afex/pbkrtest; estimates
# and confidence intervals come from emmeans with Kenward-Roger df.

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

if (!requireNamespace("lme4", quietly = TRUE)) {
  stop("Package 'lme4' is required for this analysis.")
}
if (!requireNamespace("afex", quietly = TRUE)) {
  stop("Package 'afex' is required for Kenward-Roger Type 3 F-tests.")
}
if (!requireNamespace("emmeans", quietly = TRUE)) {
  stop("Package 'emmeans' is required for Kenward-Roger contrasts and confidence intervals.")
}
if (!requireNamespace("pbkrtest", quietly = TRUE)) {
  stop("Package 'pbkrtest' is required for Kenward-Roger p values.")
}

files <- list.files(
  ratings_dir,
  pattern = "_word_depth_scores_wide\\.csv$",
  full.names = TRUE
)
files <- files[!grepl("^all_local_llm_", basename(files))]
if (length(files) < 2) stop("Need at least two model-specific wide CSV files.")

metadata_cols <- c(
  "word",
  "source_tables",
  "source_memberships",
  "roles",
  "lexical_types",
  "syntactic_frames",
  "simple_word_group",
  "simple_claim_group",
  "self_object_word_group",
  "self_object_claim_group",
  "cova_word_groups",
  "cova_claim_group",
  "cova_dimensions",
  "cova_polarities",
  "abstraction_word_groups",
  "abstraction_levels",
  "abstraction_min_level",
  "abstraction_max_level",
  "overall_claim_group",
  "any_depth_related_claim",
  "any_control_claim",
  "any_intermediate_claim"
)

detect_prefix <- function(names_vec) {
  composite_cols <- grep("_psychological_depth_composite$", names_vec, value = TRUE)
  if (length(composite_cols) != 1) {
    stop("Could not detect a single composite column in file.")
  }
  sub("_psychological_depth_composite$", "", composite_cols[[1]])
}

read_model_file <- function(path) {
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  prefix <- detect_prefix(names(dat))
  rating_model_col <- paste0(prefix, "_rating_model")
  if (!rating_model_col %in% names(dat)) stop("Missing rating model column in ", path)

  score_map <- c(
    psychological_depth_raw = paste0(prefix, "_psychological_depth_raw"),
    self_reflective_relevance_raw = paste0(prefix, "_self_reflective_relevance_raw"),
    existential_meaning_raw = paste0(prefix, "_existential_meaning_raw"),
    reflective_cognitive_process_raw = paste0(prefix, "_reflective_cognitive_process_raw"),
    personal_emotional_salience_raw = paste0(prefix, "_personal_emotional_salience_raw"),
    concrete_observable_reversed_raw = paste0(prefix, "_concrete_observable_reversed_raw"),
    logistical_transactional_reversed_raw = paste0(prefix, "_logistical_transactional_reversed_raw"),
    impersonal_neutral_reversed_raw = paste0(prefix, "_impersonal_neutral_reversed_raw"),
    concrete_observable_depth_keyed = paste0(prefix, "_concrete_observable_depth_keyed"),
    logistical_transactional_depth_keyed = paste0(prefix, "_logistical_transactional_depth_keyed"),
    impersonal_neutral_depth_keyed = paste0(prefix, "_impersonal_neutral_depth_keyed"),
    psychological_depth_composite = paste0(prefix, "_psychological_depth_composite")
  )
  missing <- setdiff(score_map, names(dat))
  if (length(missing) > 0) stop("Missing score columns in ", path, ": ", paste(missing, collapse = ", "))

  long <- dat[, metadata_cols, drop = FALSE]
  long$rating_model <- dat[[rating_model_col]]
  long$model_prefix <- prefix
  long$source_file <- basename(path)
  for (target in names(score_map)) {
    long[[target]] <- suppressWarnings(as.numeric(dat[[score_map[[target]]]]))
  }
  long
}

long_dat <- do.call(rbind, lapply(files, read_model_file))
long_dat$word <- factor(long_dat$word)
long_dat$rating_model <- factor(long_dat$rating_model)
long_dat$model_prefix <- factor(long_dat$model_prefix)
long_dat$abstraction_max_level <- suppressWarnings(as.numeric(long_dat$abstraction_max_level))

wide_by_model <- Reduce(
  function(left, right) merge(left, right, by = metadata_cols, all = TRUE, sort = TRUE),
  lapply(files, function(path) {
    dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    keep <- c(metadata_cols, grep("^[A-Za-z0-9_]+_", names(dat), value = TRUE))
    dat[, unique(keep), drop = FALSE]
  })
)

write.csv(
  long_dat,
  file.path(ratings_dir, "all_local_llm_word_depth_scores_long.csv"),
  row.names = FALSE
)
write.csv(
  wide_by_model,
  file.path(ratings_dir, "all_local_llm_word_depth_scores_wide.csv"),
  row.names = FALSE
)

fit_binary_model <- function(data, group_col, label) {
  dat <- data[data[[group_col]] %in% c("control", "depth_related"), , drop = FALSE]
  dat <- dat[is.finite(dat$psychological_depth_composite), , drop = FALSE]
  dat$depth_related_num <- ifelse(dat[[group_col]] == "depth_related", 1, 0)
  dat$depth_related_factor <- factor(dat[[group_col]], levels = c("control", "depth_related"))
  fit <- afex::mixed(
    psychological_depth_composite ~ depth_related_factor +
      (1 + depth_related_factor || rating_model) +
      (1 | word),
    data = dat,
    type = 3,
    method = "KR",
    test_intercept = TRUE,
    expand_re = TRUE,
    check_contrasts = FALSE,
    control = lme4::lmerControl(
      optimizer = "bobyqa",
      calc.derivs = TRUE,
      optCtrl = list(maxfun = 1000000)
    )
  )
  group_means <- aggregate(
    psychological_depth_composite ~ rating_model + depth_related_factor,
    data = dat,
    FUN = mean
  )
  wide_means <- reshape(
    group_means,
    idvar = "rating_model",
    timevar = "depth_related_factor",
    direction = "wide"
  )
  names(wide_means) <- sub("psychological_depth_composite\\.", "mean_", names(wide_means))
  wide_means$check <- label
  if ("mean_depth_related" %in% names(wide_means) && "mean_control" %in% names(wide_means)) {
    wide_means$difference_depth_minus_control <- wide_means$mean_depth_related - wide_means$mean_control
  }
  list(label = label, data = dat, fit = fit, per_model = wide_means)
}

fit_gradient_model <- function(data) {
  dat <- data[is.finite(data$abstraction_max_level) & is.finite(data$psychological_depth_composite), , drop = FALSE]
  fit <- afex::mixed(
    psychological_depth_composite ~ abstraction_max_level +
      (1 + abstraction_max_level || rating_model) +
      (1 | word),
    data = dat,
    type = 3,
    method = "KR",
    test_intercept = TRUE,
    expand_re = TRUE,
    check_contrasts = FALSE,
    control = lme4::lmerControl(
      optimizer = "bobyqa",
      calc.derivs = TRUE,
      optCtrl = list(maxfun = 1000000)
    )
  )
  group_means <- aggregate(
    psychological_depth_composite ~ rating_model + abstraction_max_level,
    data = dat,
    FUN = mean
  )
  list(data = dat, fit = fit, per_model = group_means)
}

extract_estimate <- function(fit, kind) {
  if (kind == "binary") {
    emm <- emmeans::emmeans(
      fit$full_model,
      ~ depth_related_factor,
      lmer.df = "kenward-roger"
    )
    est <- as.data.frame(summary(
      emmeans::contrast(emm, "trt.vs.ctrl", ref = "control"),
      infer = c(TRUE, TRUE)
    ))
    estimate_col <- "estimate"
  } else if (kind == "gradient") {
    trend <- emmeans::emtrends(
      fit$full_model,
      ~ 1,
      var = "abstraction_max_level",
      lmer.df = "kenward-roger"
    )
    est <- as.data.frame(summary(trend, infer = c(TRUE, TRUE)))
    estimate_col <- grep("\\.trend$", names(est), value = TRUE)[1]
  } else {
    stop("Unknown estimate kind: ", kind)
  }
  if (is.na(estimate_col) || !estimate_col %in% names(est)) {
    stop("Could not identify estimate column for ", kind)
  }
  list(
    estimate = est[[estimate_col]][1],
    std_error = est$SE[1],
    df_kr = est$df[1],
    ci_lower_kr = est$lower.CL[1],
    ci_upper_kr = est$upper.CL[1],
    t_value = est$t.ratio[1],
    contrast_p_value_kr = est$p.value[1]
  )
}

coef_row <- function(fit, term, check, n_obs, n_words, n_models, estimate_kind) {
  anova_tab <- as.data.frame(anova(fit))
  if (!term %in% rownames(anova_tab)) stop("Term not found in afex ANOVA table: ", term)
  test <- anova_tab[term, , drop = FALSE]
  est <- extract_estimate(fit, estimate_kind)
  data.frame(
    check = check,
    term = term,
    estimate = est$estimate,
    std_error = est$std_error,
    ci_lower_kr = est$ci_lower_kr,
    ci_upper_kr = est$ci_upper_kr,
    df_kr = est$df_kr,
    t_value = est$t_value,
    contrast_p_value_kr = est$contrast_p_value_kr,
    num_df_kr = test[["num Df"]][1],
    den_df_kr = test[["den Df"]][1],
    f_value_kr = test[["F"]][1],
    p_value_kr = test[["Pr(>F)"]][1],
    ddf_method = "Kenward-Roger",
    test_method = "afex Type 3 conditional F-test",
    n_obs = n_obs,
    n_words = n_words,
    n_models = n_models,
    singular = lme4::isSingular(fit$full_model),
    stringsAsFactors = FALSE
  )
}

binary_models <- list(
  reflective_vs_practical_words = fit_binary_model(long_dat, "simple_claim_group", "reflective_vs_practical_words"),
  inner_life_vs_object_words = fit_binary_model(long_dat, "self_object_claim_group", "inner_life_vs_object_words"),
  cova_depth_vs_control_words = fit_binary_model(long_dat, "cova_claim_group", "cova_depth_vs_control_words"),
  clean_overall_depth_vs_control_words = fit_binary_model(long_dat, "overall_claim_group", "clean_overall_depth_vs_control_words")
)
gradient_model <- fit_gradient_model(long_dat)

fixed_effects <- do.call(
  rbind,
  c(
    lapply(binary_models, function(x) {
      coef_row(
        x$fit,
        "depth_related_factor",
        x$label,
        nrow(x$data),
        length(unique(x$data$word)),
        length(unique(x$data$rating_model)),
        "binary"
      )
    }),
    list(
      coef_row(
        gradient_model$fit,
        "abstraction_max_level",
        "abstraction_gradient_ordering",
        nrow(gradient_model$data),
        length(unique(gradient_model$data$word)),
        length(unique(gradient_model$data$rating_model)),
        "gradient"
      )
    )
  )
)

per_model_binary <- do.call(rbind, lapply(binary_models, function(x) x$per_model))
per_model_gradient <- gradient_model$per_model

model_counts <- aggregate(
  psychological_depth_composite ~ rating_model,
  data = long_dat,
  FUN = length
)
names(model_counts)[2] <- "n_word_ratings"

write.csv(
  fixed_effects,
  file.path(output_dir, "all_local_llm_lme4_fixed_effects.csv"),
  row.names = FALSE
)
write.csv(
  per_model_binary,
  file.path(output_dir, "all_local_llm_per_model_binary_means.csv"),
  row.names = FALSE
)
write.csv(
  per_model_gradient,
  file.path(output_dir, "all_local_llm_per_model_abstraction_means.csv"),
  row.names = FALSE
)
write.csv(
  model_counts,
  file.path(output_dir, "all_local_llm_model_counts.csv"),
  row.names = FALSE
)

summary_path <- file.path(report_dir, "all_local_llm_lme4_model_summaries.txt")
sink(summary_path)
cat("MULTI-MODEL LOCAL LLM SEMANTIC AUDIT WITH LME4\n")
cat("================================================\n\n")
cat("Fixed-effect p values use Type 3 conditional F-tests with Kenward-Roger approximation via afex/pbkrtest.\n")
cat("Contrasts, slopes, and confidence intervals use emmeans with Kenward-Roger df.\n\n")
cat("Input model files:\n")
print(basename(files))
cat("\nLong observations: ", nrow(long_dat), "\n", sep = "")
cat("Unique words: ", length(unique(long_dat$word)), "\n", sep = "")
cat("Rating models: ", paste(levels(long_dat$rating_model), collapse = ", "), "\n\n", sep = "")
for (name in names(binary_models)) {
  cat("\nMODEL: ", name, "\n", sep = "")
  cat("Formula: psychological_depth_composite ~ depth_related_factor + (1 + depth_related_factor || rating_model) + (1 | word)\n")
  print(anova(binary_models[[name]]$fit))
}
cat("\nMODEL: abstraction_gradient_ordering\n")
cat("Formula: psychological_depth_composite ~ abstraction_max_level + (1 + abstraction_max_level || rating_model) + (1 | word)\n")
print(anova(gradient_model$fit))
sink()

fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
fmt_p <- function(x) {
  ifelse(
    is.na(x),
    "NA",
    ifelse(x < .001, "< .001", sub("^0", "", formatC(x, format = "f", digits = 3)))
  )
}
fmt_p_text <- function(x) {
  ifelse(is.na(x), "p = NA", ifelse(x < .001, "p < .001", paste0("p = ", fmt_p(x))))
}
fmt_df <- function(x) {
  ifelse(
    is.na(x),
    "NA",
    ifelse(abs(x - round(x)) < .001, formatC(x, format = "f", digits = 0), formatC(x, format = "f", digits = 2))
  )
}

report <- c(
  "# All-local-LLM post hoc semantic stimulus audit",
  "",
  paste0("- Models: ", paste(levels(long_dat$rating_model), collapse = ", ")),
  paste0("- Long observations: ", nrow(long_dat)),
  paste0("- Unique words: ", length(unique(long_dat$word))),
  paste0("- Model formula, binary contrasts: `psychological_depth_composite ~ depth_related_factor + (1 + depth_related_factor || rating_model) + (1 | word)`"),
  paste0("- Model formula, abstraction gradient: `psychological_depth_composite ~ abstraction_max_level + (1 + abstraction_max_level || rating_model) + (1 | word)`"),
  "- Fixed-effect p values use Type 3 conditional F-tests with Kenward-Roger approximation via `afex`/`pbkrtest`.",
  "- Contrasts, slopes, and confidence intervals use `emmeans` with Kenward-Roger df.",
  "",
  "## Fixed effects",
  ""
)
for (i in seq_len(nrow(fixed_effects))) {
  row <- fixed_effects[i, ]
  report <- c(
    report,
    paste0(
      "- ", row$check,
      ": estimate = ", fmt(row$estimate),
      ", SE = ", fmt(row$std_error),
      ", 95% CI [", fmt(row$ci_lower_kr), ", ", fmt(row$ci_upper_kr), "]",
      ", F(", fmt_df(row$num_df_kr), ", ", fmt_df(row$den_df_kr), ") = ", fmt(row$f_value_kr),
      ", ", fmt_p_text(row$p_value_kr),
      ", observations = ", row$n_obs,
      ", words = ", row$n_words,
      ", models = ", row$n_models,
      ", singular = ", row$singular
    )
  )
}
report <- c(
  report,
  "",
  "## Interpretation boundary",
  "",
  "These are post hoc LLM-based semantic stimulus-audit results. They assess whether local LLMs rate the analyzed target words as semantically deeper than control words. They should not be described as human construct validation.",
  ""
)
writeLines(report, file.path(report_dir, "all_local_llm_lme4_validation_report.md"))

cat("Wrote combined datasets and lme4 outputs to ", output_dir, "\n", sep = "")
cat("Summary file: ", summary_path, "\n", sep = "")

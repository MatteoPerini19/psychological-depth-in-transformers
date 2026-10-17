#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
})

root <- normalizePath(".", mustWork = TRUE)
out_dir <- file.path(root, "figures", "main")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

model_levels <- c(
  "gpt2",
  "distilgpt2",
  "pythia-14m",
  "pythia-70m",
  "pythia-70m-deduped",
  "pythia-160m"
)

model_labels <- c(
  "gpt2" = "GPT-2",
  "distilgpt2" = "DistilGPT-2",
  "pythia-14m" = "Pythia-14M",
  "pythia-70m" = "Pythia-70M",
  "pythia-70m-deduped" = "Pythia-70M deduped",
  "pythia-160m" = "Pythia-160M"
)

norm_model <- function(x) {
  x <- sub("^EleutherAI/", "", x)
  factor(x, levels = model_levels, labels = unname(model_labels[model_levels]))
}

theme_manuscript <- function(base_size = 10) {
  theme_classic(base_size = base_size, base_family = "serif") +
    theme(
      axis.text = element_text(color = "black"),
      axis.title = element_text(color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.35),
      axis.ticks = element_line(color = "black", linewidth = 0.35),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.key = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", color = "black"),
      plot.margin = margin(7, 7, 7, 7)
    )
}

save_both <- function(plot, stem, width, height) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, units = "in", dpi = 450, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, units = "in", device = "pdf", bg = "white")
}

layer_path <- file.path(
  root,
  "logit_lens_depth_mini",
  "paper_sentence_completion_tests",
  "outputs",
  "cross_model_sentence_completion_summary_by_layer.csv"
)
model_path <- file.path(
  root,
  "logit_lens_depth_mini",
  "paper_sentence_completion_tests",
  "outputs",
  "cross_model_sentence_completion_summary_by_model.csv"
)
patch_path <- file.path(
  root,
  "logit_lens_depth_mini",
  "paper_sentence_completion_tests",
  "results",
  "robustness",
  "patching_bootstrap.csv"
)

# Figure 1: primary self/object layerwise trajectories.
layer_df <- read_csv(layer_path, show_col_types = FALSE) %>%
  filter(probe_name == "self_object", stem_condition %in% c("self", "object")) %>%
  mutate(
    model = norm_model(model_name),
    condition = recode(
      stem_condition,
      self = "Self-directed stems",
      object = "Object-directed stems"
    )
  )

window_df <- layer_df %>%
  distinct(model, n_transformer_layers) %>%
  mutate(
    middle_start = if_else(n_transformer_layers == 12, 3 / 12, 2 / 6),
    middle_end = if_else(n_transformer_layers == 12, 6 / 12, 3 / 6),
    late_start = if_else(n_transformer_layers == 12, 8 / 12, 4 / 6),
    late_end = if_else(n_transformer_layers == 12, 11 / 12, 5 / 6)
  )

fig1 <- ggplot(layer_df, aes(x = relative_depth, y = mean_logit_margin, color = condition, linetype = condition)) +
  geom_rect(
    data = window_df,
    aes(xmin = middle_start, xmax = middle_end, ymin = -Inf, ymax = Inf),
    inherit.aes = FALSE,
    fill = "grey92",
    color = NA
  ) +
  geom_rect(
    data = window_df,
    aes(xmin = late_start, xmax = late_end, ymin = -Inf, ymax = Inf),
    inherit.aes = FALSE,
    fill = "#DCEBFA",
    color = NA
  ) +
  geom_hline(yintercept = 0, linewidth = 0.3, color = "grey35") +
  geom_line(linewidth = 0.75) +
  geom_point(size = 1.15) +
  facet_wrap(~ model, ncol = 3) +
  scale_color_manual(values = c("Self-directed stems" = "#2166AC", "Object-directed stems" = "black")) +
  scale_linetype_manual(values = c("Self-directed stems" = "solid", "Object-directed stems" = "dashed")) +
  scale_x_continuous(
    breaks = c(0, 0.5, 1),
    labels = c("0", ".5", "1"),
    limits = c(0, 1)
  ) +
  labs(
    x = "Normalized depth",
    y = "Inner-life minus object/logistical logit margin"
  ) +
  theme_manuscript(base_size = 10)

save_both(fig1, "figure_1_layerwise_self_object", width = 7.2, height = 5.4)

# Figure 2: self/object advantage and depth-taxonomy late-minus-middle effects.
model_df <- read_csv(model_path, show_col_types = FALSE) %>%
  filter(
    (probe_name == "self_object_advantage" & stem_condition == "self_minus_object_stems") |
      (probe_name == "cova_verb_noun" & stem_condition == "all_stems")
  ) %>%
  mutate(
    model = norm_model(model_name),
    probe = recode(
      probe_name,
      self_object_advantage = "Self/object inner-life",
      cova_verb_noun = "Depth-taxonomy verb/noun"
    ),
    se = late_minus_middle_sd / sqrt(n_stems),
    ci_low = late_minus_middle - qt(.975, late_minus_middle_df) * se,
    ci_high = late_minus_middle + qt(.975, late_minus_middle_df) * se
  )

fig2 <- ggplot(model_df, aes(x = model, y = late_minus_middle, color = probe, shape = probe)) +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "grey35") +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    position = position_dodge(width = 0.55),
    width = 0.18,
    linewidth = 0.45
  ) +
  geom_point(position = position_dodge(width = 0.55), size = 2.35) +
  scale_color_manual(values = c("Self/object inner-life" = "#2166AC", "Depth-taxonomy verb/noun" = "#2E8B57")) +
  scale_shape_manual(values = c("Self/object inner-life" = 16, "Depth-taxonomy verb/noun" = 17)) +
  labs(
    x = NULL,
    y = "Late-minus-middle logit-margin effect"
  ) +
  theme_manuscript(base_size = 10) +
  theme(axis.text.x = element_text(angle = 28, hjust = 1, vjust = 1))

save_both(fig2, "figure_2_primary_and_cova_effects", width = 7.2, height = 4.4)

# Figure 3: residual-stream patching, forward and reverse.
patch_df <- read_csv(patch_path, show_col_types = FALSE) %>%
  filter(
    component == "resid_post",
    control_type %in% c("matched_forward_self_to_object", "reverse_object_to_self")
  ) %>%
  mutate(
    model = norm_model(model_name),
    direction = recode(
      control_type,
      matched_forward_self_to_object = "Forward self-to-object",
      reverse_object_to_self = "Reverse object-to-self"
    )
  )

fig3 <- ggplot(patch_df, aes(x = model, y = late_mean_delta, color = direction, shape = direction)) +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "grey35") +
  geom_errorbar(
    aes(ymin = bootstrap_ci_low, ymax = bootstrap_ci_high),
    position = position_dodge(width = 0.55),
    width = 0.18,
    linewidth = 0.45
  ) +
  geom_point(position = position_dodge(width = 0.55), size = 2.35) +
  scale_color_manual(values = c("Forward self-to-object" = "#2166AC", "Reverse object-to-self" = "#B2182B")) +
  scale_shape_manual(values = c("Forward self-to-object" = 16, "Reverse object-to-self" = 17)) +
  labs(
    x = NULL,
    y = "Raw delta in inner-life/object margin"
  ) +
  theme_manuscript(base_size = 10) +
  theme(axis.text.x = element_text(angle = 28, hjust = 1, vjust = 1))

save_both(fig3, "figure_3_patching_forward_reverse", width = 7.2, height = 4.4)

message("Created main figures in ", out_dir)

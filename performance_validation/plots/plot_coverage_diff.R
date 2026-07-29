library(ggplot2)
library(ggforestplot)
library(tidyr)
library(dplyr)

base_dir <- "../results/analysis_tables"

# Rename the dgps
coverage <- read.csv(file.path(base_dir, "results_coverage_ci.csv")) |>
  mutate(
    dgp = recode(
      dgp,
      "DGPBiNorm-1_1_2.0_0.5_1.0" = "BiNorm",
      "DGPExp_1" = "Exp",
      "DGPLogNorm_0_1" = "LogNorm",
      "DGPNorm_0_1" = "Norm"
    ),
    n = factor(n),
    # Add an ID that carries DGP and statistic info
    id = interaction(
      dgp,
      statistic,
      drop = TRUE
    ),
    id = factor(
      id,
      levels = c(
        "BiNorm.corr",
        "LogNorm.median",
        "LogNorm.mean",
        "Exp.median",
        "Exp.mean",
        "Norm.median",
        "Norm.mean"
      )
    )
  ) |> filter(
    method == "double"
  ) |>
  rename(
    # Rename this for easier capturing
    lower_coverage_diff_estimate = lower_coverage_diff,
    upper_coverage_diff_estimate = upper_coverage_diff,
    two_sided_coverage_diff_estimate = two_sided_coverage_diff
  ) |>
  pivot_longer(
    cols = matches(
      "^(lower|upper|two_sided)_coverage_diff_"
    ),
    names_to = c("bound", ".value"),
    names_pattern = 
      "^(lower|upper|two_sided)_coverage_diff_(estimate|CI_low|CI_high)$"
  ) |>
  mutate(
    bound = factor(
      bound,
      levels = c(
        "lower",
        "upper",
        "two_sided"
      )
    )
  )


plot <- ggplot(coverage, aes(
  # Use percentages
  x = estimate * 100,
  y = id
  )
) +
  geom_stripes() +
  geom_vline(
    xintercept = 0,
    linetype = "solid",
    linewidth = 0.5,
    colour = "black"
  ) +
  geom_effect(
    aes(
      xmin = CI_low * 100,
      xmax = CI_high * 100,
      colour = n
    ),
    position = ggstance::position_dodgev(
      height = 0.5
    )
  ) +
  facet_wrap(
    ~ bound,
    nrow = 1,
    labeller = as_labeller(c(
      lower = "Enostranski spodnji interval",
      upper = "Enostranski zgornji interval",
      two_sided = "Dvostranski interval"
    ))
  ) +
  scale_y_discrete(
    labels = \(x) gsub("\\.", " ", x)
  ) +
  labs(
    colour = "Velikost vzorca",
    x = "Razlika v pokritju v odstotnih točkah (naša - referenčna)",
    y = NULL
  ) +
  theme_forest(base_size = 12) +
    theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 12)
  )

output_dir <- "figures"

ggsave(
  filename = file.path(output_dir, "coverage_diff_CI.pdf"),
  device = cairo_pdf,
  plot = plot,
)
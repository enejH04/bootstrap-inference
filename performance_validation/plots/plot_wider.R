library(ggplot2)
library(ggforestplot)
library(tidyr)
library(dplyr)

base_dir <- "../results/analysis_tables"

wider <- read.csv(file.path(base_dir, "results_wider.csv")) |>
  rename(
    # Rename clumsy name
    estimate = Pr.our_wider..Pr.ref_wider.
  ) |>
  filter(
    method == "double"
  ) |>
  mutate(
    # Rename the dgps and make id field
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
  )

plot <- ggplot(wider, aes(
  x = estimate * 100,
  y = id
)) +
  geom_stripes(
    even = "#80808018",
    odd = "#FFFFFF00"
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "solid",
    linewidth = 0.3,
    colour = "black"
  ) +
  geom_effect(
    aes(
      xmin = wider_CI_low * 100,
      xmax = wider_CI_high * 100,
      colour = n,
      shape = n
    ),
    position = ggstance::position_dodgev(
      height = 0.5
    ),
    size = 0.35
  ) +
  scale_x_continuous(
    limits = c(-13, 13),
    breaks = seq(-10, 10, 2.5)
  ) +
  scale_y_discrete(
    labels = \(x) gsub("\\.", " - ", x)
  ) +
  labs(
    colour = "Velikost vzorca",
    shape = "Velikost vzorca",
    x = "Razlika med verjetnostma v odstotnih točkah (naša - referenčna)",
    y = NULL
  ) +
  theme_forest() +
  theme(
    # These text settings seem to work well
    text = element_text(
      size = 6,
      family = "Helvetica"
    ),
    
    axis.text.x = element_text(size = 6),
    axis.text.y = element_text(size = 6),
    axis.title.x = element_text(size = 6),
    
    strip.text = element_text(
      size = 6,
      face = "bold"
    ),
    
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 6),
    legend.margin = margin(t = -8),
    legend.key.spacing = grid::unit(1, "mm")
  ) +
  guides(
    colour = guide_legend(
      override.aes = list(size = 1.0)
    )
  )

# Dimensions for the thesis
aspect_ratio <- 16 / 14 
pt_to_mm <- 0.35146
column_w_pts <- 353.40038

plot_w_mm <- column_w_pts * pt_to_mm / 1.5
plot_h_mm <- plot_w_mm / aspect_ratio

output_dir <- "figures"

# Save figure on mac without rendering issues
ggsave(
  filename = file.path(output_dir, "wider_diff_CI.pdf"),
  plot = plot,
  width = plot_w_mm,
  height = plot_h_mm,
  units = "mm",
  device = function(filename, width, height, ...) {
    grDevices::quartz(
      file = filename,
      type = "pdf",
      width = width,
      height = height,
      family = "Helvetica"
    )
  }
)
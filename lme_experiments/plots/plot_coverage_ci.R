library(ggplot2)
library(tidyr)
library(dplyr)

base_dir <- "../results/analysis_tables"

l <- 95

df <- read.csv(file.path(base_dir, "results_coverage_ci.csv")) |>
  pivot_longer(
    cols = matches(
      "^coverage_(50|90|95)_"
    ),
    names_to = c("level", ".value"),
    names_pattern = "coverage_(50|90|95)_(estimate|ci_low|ci_high)" 
  ) |>
  mutate(
    level = as.integer(level),
    n_l3 = factor(n_l3, levels = c(15, 50)),
    rand_eff_dgp = factor(
      rand_eff_dgp,
      levels = c("norm", "t", "lognorm"),
      labels = c("Norm", "t", "LogNorm")
    ),
    stat = factor(
      stat,
      levels = c("mu", "sd_l3", "sd_l2"),
      labels = c("mu", "sd (L3)", "sd (L2)")
    ),
    method = factor(
      method,
      levels = c(
        "profile-likelihood",
        "parametric-percentile-boot",
        "cases-percentile-boot",
        "cases-double-boot"
      ),
      labels = c(
        "PL",
        "PPB",
        "CPB",
        "CDB"
      )
    )
  ) |> 
  filter(
    level == l
  )

plot <- ggplot(
  df,
  aes(
    x = n_l3,
    y = estimate,
    colour = method,
  )
) +
  annotate(
    "rect",
    xmin = -Inf, xmax = 1.5,
    ymin = -Inf, ymax = Inf,
    fill = "#80808018",
  ) +
  geom_hline(
    yintercept = l / 100,
    colour = "black",
    linewidth = 0.4
  ) + 
  scale_y_continuous(
    breaks = c(0.40, 0.60, 0.80, 0.95, 1.00)
  ) +
  facet_grid(
    rows = vars(stat),
    cols = vars(rand_eff_dgp)
  ) + 
  geom_point(
    position = position_dodge(0.5),
    size = 0.55
  ) +
  geom_errorbar(
    aes(
      ymin = ci_low,
      ymax = ci_high
    ),
    width = 0,
    linewidth = 0.45,
    position = position_dodge(0.5)
  ) +
  labs(
    x = "Število skupin na tretjem nivoju",
    y = "Verjetnost pokritja",
    colour = "Metoda"
  ) +
  theme_bw() +
  theme(
    strip.background = element_blank(),
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
aspect_ratio <- 1
pt_to_mm <- 0.35146
column_w_pts <- 353.40038

plot_w_mm <- column_w_pts * pt_to_mm
plot_h_mm <- plot_w_mm / aspect_ratio

output_dir <- "figures"

# Save figure on mac without rendering issues
ggsave(
  filename = file.path(output_dir, sprintf("hierarch_coverage_%d.pdf", l)),
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
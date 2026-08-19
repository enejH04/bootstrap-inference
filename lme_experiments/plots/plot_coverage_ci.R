library(ggplot2)
library(tidyr)
library(dplyr)

base_dir <- "../results/analysis_tables"

l <- 50

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
    stat = factor(stat, levels = c("mu", "sd_l3", "sd_l2")),
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
    shape = method,
  )
) +
  geom_hline(
    yintercept = l / 100,
    colour = "grey40"
  ) + 
facet_grid(
  rows = vars(stat),
  cols = vars(rand_eff_dgp)
) + 
  geom_point(
    position = position_dodge(0.5)
  ) +
  geom_errorbar(
    aes(
      ymin = ci_low,
      ymax = ci_high
    ),
    width = 0,
    position = position_dodge(0.5)
  ) +
  labs(
    title = l
  )
  theme_bw()
  
output_dir <- "figures"

ggsave(
  filename = file.path(output_dir, sprintf("hierarch_coverage_%d.pdf", l)),
  plot = plot
)
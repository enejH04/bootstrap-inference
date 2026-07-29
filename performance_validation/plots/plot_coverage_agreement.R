library(ggplot2)
library(tidyr)
library(dplyr)

# Relative to the location of the .Rproj file
base_dir <- "../results/analysis_tables"

coverage_summary <- read.csv(file.path(base_dir, "results_coverage_summary.csv"))

# Convert the coverage summary from a wide to a long table (in order for ggplot
# to be able to use coverage_type to create one panel per type)
coverage_long <- coverage_summary %>%
  pivot_longer(
    cols = c(
      lower_coverage_our,
      lower_coverage_ref,
      upper_coverage_our,
      upper_coverage_ref,
      two_sided_coverage_our,
      two_sided_coverage_ref
    ),
  names_to = c("coverage_type", "library"),
  # Define capture groups
  names_pattern = "(lower|upper|two_sided)_coverage_(our|ref)",
  values_to = "coverage"
) %>%
  pivot_wider(
    names_from = library,
    values_from = coverage
  )

coverage_double <- coverage_long %>% filter(method == "double")

# factor tells R that a column represents categories with a defined set and
# order. Important for plotting in the desired order
coverage_double$coverage_type <- factor(
  coverage_double$coverage_type,
  levels = c("lower", "upper", "two_sided")
)

coverage_agreement_plot <- ggplot(
  coverage_double,
  aes(x = ref, y = our)
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = "grey50"
  ) +
  geom_point(
    size = 2.5,
    alpha = 0.6,
    colour = "steelblue"
  ) + facet_wrap(
    ~coverage_type,
    labeller = as_labeller(c(
      lower = "Enostranski spodnji interval",
      upper = "Enostranski zgornji interval",
      two_sided = "Dvostranski interval"
    ))
  ) +
  coord_equal(xlim = c(0.8, 1), ylim = c(0.8, 1)) +
  labs(
    x = "Verjetnost pokritja referenčne knjižnice",
    y = "Verjetnost pokritja naše knjižnice",
  ) +
  theme_bw(
    base_size = 10,
    base_family = "Helvetica"
  ) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 12),
  )

output_dir <- "figures"

ggsave(
  filename = file.path(output_dir, "coverage_agreement.pdf"),
  device = cairo_pdf,
  plot = coverage_agreement_plot,
)

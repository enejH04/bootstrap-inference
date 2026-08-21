# Simulation study using hierarchical data

This directory contains all the code used in the simulation study in which we compare different methods for constructing confidence intervals
for parameters of a random-effects model under violations of the normality assumption (section 4 of the thesis). 

The full analysis is available in the [notebook](analysis.ipynb). We also provide our [raw results](results/results.csv) used in the analysis so that the computationally expensive simulation can be skipped.

## Reproduction of the results

To reproduce the results from the thesis, we recommend using [uv](https://docs.astral.sh/uv/).

This experiment also requires an active installation of [R](https://www.r-project.org/). For the simulation study, we used version 4.6.1. For fitting linear mixed effects models we used the [lme4](https://cran.r-project.org/web/packages/lme4/index.html) package version 2.0.6.

The experiment stores `results.csv` and `config.json` in `lme_experiments/RESULTS_DIR`. Existing compatible results are resumed. Use a new directory when changing the experiment configuration.

Note that the full simulation may take several days depending on the available CPU resources. The exact numerical results may differ slightly across operating systems, hardware and software versions due to differences in floating-point arithmetic and execution. These differences should be minor and should not affect the conclusions of the analysis.

Run the full experiment from the repository root:

```bash
# Run from repository root
uv run --frozen --group lme-experiments python -m lme_experiments.run_experiment RESULTS_DIR
```

## Reproduction of the figures

All the code used in creating the figures is available in the [plots](plots) directory.
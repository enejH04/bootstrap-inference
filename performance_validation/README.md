# Empirical validation of the double percentile implementation

This directory contains all the code used in the simulation study in which we compare our implementation with the reference implementation provided by **bootstrap-ci** (section 3 of the thesis). 

The full analysis is available in the [notebook](analysis.ipynb). We also provide our [raw results](results/results.csv) used in the analysis so that the computationally expensive simulation can be skipped.

## Reproduction of the results

To reproduce the results from the thesis, we recommend using [uv](https://docs.astral.sh/uv/).

The experiment writes `performance_validation/results/results.csv`, replacing any existing file at that location.

Note that the full simulation may take several hours depending on the available CPU resources. The exact numerical results may differ slightly across operating systems and hardware due to differences in floating-point arithmetic and execution. These differences should be minor and should not affect the conclusions of the analysis.

Run the full experiment from the repository root:

```bash
# Run from repository root
uv run --frozen --group performance-validation python -m performance_validation.comparison
```

## Reproduction of the figures

All the code used in creating the figures is available in the [plots](plots) directory.
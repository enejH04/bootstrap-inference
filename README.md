# bootstrap-inference

Extensible bootstrap inference in Python.

**bootstrap-inference** provides percentile and double-percentile bootstrap
confidence intervals, along with bootstrap estimates of bias and variance. It
supports arbitrary user-defined statistics and several resampling strategies:

- IID resampling
- Hierarchical resampling using the cases bootstrap
- Moving, circular, non-overlapping, and stationary block resampling for time series

The library was developed as part of a BSc thesis at the University of
Ljubljana, Faculty of Computer and Information Science (UL FRI), under the
supervision of Prof. Dr. Erik Štrumbelj.

- [Documentation](https://bootstrap-inference.readthedocs.io)
- [Source code](https://github.com/enejH04/bootstrap-inference)
- [Performance validation experiments](https://github.com/enejH04/bootstrap-inference/tree/main/performance_validation)
- [Hierarchical data experiments](https://github.com/enejH04/bootstrap-inference/tree/main/lme_experiments)

## Installation

**bootstrap-inference** requires Python 3.12 or later. Install it from PyPI with:

```bash
pip install bootstrap-inference
```

## A quick example

This example estimates the sample mean and constructs a two-sided 95%
double-bootstrap confidence interval using IID resampling.

First, create a sample and initialize the bootstrap procedure:

```python
import numpy as np
from bootstrap_inference import Bootstrap, IIDResampler

rng = np.random.default_rng(42)

# Generate 32 IID draws from a standard normal distribution
sample = rng.standard_normal(32)

resampler = IIDResampler(data_sample=sample)
boot = Bootstrap(
    statistic=np.mean,
    resampler=resampler,
)
```

Then compute the double-percentile bootstrap confidence interval, splitting the workload
across five processes using the `n_jobs=5` parameter:

```python
ci = boot.double_percentile_ci(
    confidence_level=0.95,
    side="two",
    b1_resamples=1_000,
    b2_resamples=1_000,
    n_jobs=5,
    seed=42,
)

print(ci)
```

With the given seed, the output is:

```txt
estimate = 0.06998925652242104
lower = -0.27410311445480895
upper = 0.400736328783347
confidence level = 0.95
side = two
```

## License

This project is licensed under the MIT License.

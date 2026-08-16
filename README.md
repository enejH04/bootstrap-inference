# double-boot

Extensible double bootstrap inference in Python.

This library is the result of a BSc thesis completed at the University of Ljubljana, Faculty of Computer and Information Science (UL FRI), under the supervision of Prof. Dr. Erik Štrumbelj.

- [Documentation](https://double-boot.readthedocs.io/en/latest/)
- [Performance validation experiments](./performance_validation/)
- [Hierarchical data experiments](./lme_experiments/)

## A quick example

This example estimates the sample mean and constructs a two-sided 95%
double-bootstrap confidence interval using IID resampling.


First, create a sample and initialize the bootstrap procedure:

```python
import numpy as np
from double_boot import Bootstrap, IIDResampler

rng = np.random.default_rng(42)

# Generate 32 IID draws from a standard normal distribution
sample = rng.standard_normal(32)

resampler = IIDResampler(data_sample=sample)
boot = Bootstrap(
    statistic=np.mean,
    resampler=resampler,
)
```

Then compute the double-bootstrap confidence interval, splitting the workload
across five processes:

```python
ci = boot.double_percentile_ci(
    confidence_level=0.95,
    side="two",
    b1_resamples=1_000,
    b2_resamples=1_000,
    n_jobs=5,
    seed=42,
)
```

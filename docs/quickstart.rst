Quickstart
==========

This example estimates the sample mean and constructs a two-sided 95% bootstrap
confidence interval for it.

First, create a sample and select the IID resampling strategy:

.. code-block:: python

   import numpy as np

   from double_boot import Bootstrap, IIDResampler

   data = np.array([2.1, 2.5, 2.7, 3.2, 3.5, 3.8, 4.1, 4.4])
   resampler = IIDResampler(data)
   bootstrap = Bootstrap(np.mean, resampler)

Percentile confidence interval
------------------------------

Use :meth:`~double_boot.Bootstrap.percentile_ci` for an ordinary percentile
bootstrap interval:

.. code-block:: python

   interval = bootstrap.percentile_ci(
       confidence_level=0.95,
       b_resamples=1_000,
       seed=42,
   )

   print(interval.estimate)
   print(interval.lower, interval.upper)

Setting a seed makes the result reproducible. Increase ``b_resamples`` when
more precise bootstrap inference is required.

Double-bootstrap confidence interval
------------------------------------

Use :meth:`~double_boot.Bootstrap.double_percentile_ci` to calibrate the
interval with a second level of resampling:

.. code-block:: python

   interval = bootstrap.double_percentile_ci(
       confidence_level=0.95,
       b1_resamples=1_000,
       b2_resamples=1_000,
       n_jobs=5,
       seed=42,
   )

   print(interval)

The double bootstrap is more computationally expensive because each
first-level resample produces a second set of bootstrap resamples. Therefore
we allow the simulation to run within 5 processes by setting ``n_jobs = 5``.

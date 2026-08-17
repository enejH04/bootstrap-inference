bootstrap-inference
===================

Extensible bootstrap inference in Python.

**bootstrap-inference** provides percentile and double-percentile bootstrap
confidence intervals, along with bootstrap estimates of bias and variance. It
supports arbitrary user-defined statistics and several resampling strategies:


* IID resampling
* Hierarchical resampling using the cases bootstrap
* Moving, circular, non-overlapping, and stationary block resampling for
   time series

The library was developed as part of a BSc thesis at the University of
Ljubljana, Faculty of Computer and Information Science (UL FRI), under the
supervision of Prof. Dr. Erik Štrumbelj.

Installation
------------

**bootstrap-inference** requires Python 3.12 or later. Install it from PyPI
with:

.. code-block:: console

   pip install bootstrap-inference

Getting started
---------------

See the :doc:`quickstart` for examples of ordinary percentile and
double-percentile bootstrap confidence intervals.

The complete public interface is available in the :doc:`api` reference.

Links
-----

* `Source code <https://github.com/enejH04/bootstrap-inference>`_
* `Issue tracker <https://github.com/enejH04/bootstrap-inference/issues>`_
* `Performance validation experiments
<https://github.com/enejH04/bootstrap-inference/tree/main/performance_validation>`_
* `Hierarchical data experiments <https://github.com/enejH04/bootstrap-inference/tree/main/lme_experiments>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   quickstart
   api

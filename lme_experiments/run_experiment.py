import warnings

import numpy as np

from bootstrap_diagnostics import Bootstrap, HierarchicalResampler
from lme_experiments.generate_synthetic import generate_hierarchical_dataset
from lme_experiments.r_interface import (
    compute_parametric_percentile_ci,
    compute_profile_likelihood_cis,
    statistic,
)

# Ignore Bootstrap library DataFrame warnings about performance
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Experiment size regimes
    sizes = [
        (5, 4, 8),
        (50, 4, 8),
    ]

    rand_eff_dpgs = ["norm", "t", "lognorm"]

    # Repeat each experiment configuration 1000 times
    n_repetitions = 1000

    # Number of resamples at the top level of the bootstrap
    B = 1000
    # Number of resamples at the second level of the bootstrap
    C = 10

    # run_paired_experiment(sizes=sizes, rand_eff_dpgs=rand_eff_dpgs, )

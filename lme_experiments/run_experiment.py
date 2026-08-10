import warnings

import numpy as np

from bootstrap_diagnostics import Bootstrap, HierarchicalResampler
from lme_experiments.generate_synthetic import generate_hierarchical_dataset
from lme_experiments.r_interface import compute_profile_cis, statistic

# Ignore library DataFrame warnings about performance
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df = generate_hierarchical_dataset(50, 4, 8, seed=42)

    profile_cis = compute_profile_cis(df)

    print(profile_cis)

    # resampler = HierarchicalResampler(
    #     data_sample=df,
    #     hierarchy=[("l3", True), ("l2", True)],
    #     observation_replacement=True,
    # )
    # boot = Bootstrap(statistic, resampler)

    # ci = boot.double_percentile_ci(
    #     confidence_level=0.95,
    #     side="two",
    #     b1_resamples=1000,
    #     b2_resamples=10,
    #     n_jobs=-1,
    #     seed=52,
    # )

    # ci2 = boot.percentile_ci(
    #     confidence_level=0.95,
    #     side="two",
    #     b_resamples=1000,
    #     use_cached=True,
    # )

    # print(ci)
    # print(ci2)

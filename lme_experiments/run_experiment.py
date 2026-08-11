import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_diagnostics import (
    Bootstrap,
    ConfidenceInterval,
    HierarchicalResampler,
)
from lme_experiments.generate_synthetic import (
    EffectDist,
    generate_hierarchical_dataset,
)
from lme_experiments.r_interface import (
    CI_COLUMNS,
    STATS,
    compute_parametric_percentile_ci,
    compute_profile_likelihood_cis,
    statistic,
)

# Ignore Bootstrap library DataFrame warnings about performance
warnings.filterwarnings("ignore")

BASE_SEED = 42

# Change to desired number of cpus
N_CPUS = 16

# Resampling strategy used during the cases bootstrap
HIERARCHY = [("l3", True), ("l2", True)]

# Levels used in the simulation study
LEVELS = [0.50, 0.90, 0.95]

# Ground truth for the synthetic examples
DGP_CONFIG = {
    "mu": 0.0,
    "var_l3": 0.3,
    "var_l2": 0.3,
    "var_l1": 0.4,
}


# Store result in a dataframe that is compatible with the dataframes received
# from R
def ci_results_to_df(
    results: list[ConfidenceInterval],
    method: str,
) -> pd.DataFrame:
    ci_50, ci_90, ci_95 = results

    # Extract estimate from a single
    estimate = ci_50.estimate

    # Stack them next to eachother
    values = np.column_stack(
        [
            ci_95.lower,  # 0.025
            ci_90.lower,  # 0.050
            ci_50.lower,  # 0.250
            ci_50.upper,  # 0.750
            ci_90.upper,  # 0.950
            ci_95.upper,  # 0.975
        ]
    )

    df = pd.DataFrame(values, columns=CI_COLUMNS)
    df["estimate"] = estimate
    df["stat"] = STATS
    df["method"] = method

    return df


# Computes nonparametric percentile and double CIs
def compute_nonparametric_cases_ci(
    data_sample: pd.DataFrame,
    B: int,
    C: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resampler = HierarchicalResampler(
        data_sample,
        hierarchy=HIERARCHY,
        observation_replacement=True,
    )

    boot = Bootstrap(statistic, resampler=resampler)

    # Run the first computation to get the bootstrap distribution and cache it
    results_double = [
        boot.double_percentile_ci(
            confidence_level=LEVELS[0],
            side="two",
            b1_resamples=B,
            b2_resamples=C,
            q_est_method="median_unbiased",
            n_jobs=N_CPUS,
            seed=seed,
            use_cached=False,
        )
    ]

    results_percentile = []

    # The extra arguments aren't really needed if we use caching but we add them
    # anyway
    for level in LEVELS:
        # Don't store the cached one again
        if level != LEVELS[0]:
            results_double.append(
                boot.double_percentile_ci(
                    confidence_level=level,
                    side="two",
                    b1_resamples=B,
                    b2_resamples=C,
                    q_est_method="median_unbiased",
                    n_jobs=-1,
                    seed=seed,
                    use_cached=True,
                ),
            )

        results_percentile.append(
            boot.percentile_ci(
                confidence_level=level,
                side="two",
                b_resamples=B,
                q_est_method="median_unbiased",
                n_jobs=-1,
                seed=seed,
                use_cached=True,
            )
        )

    df_percentile = ci_results_to_df(
        results_percentile, "cases-percentile-boot"
    )
    df_double = ci_results_to_df(results_double, "cases-double-boot")

    return df_percentile, df_double


def run_paired_experiment(
    sizes: list[tuple[int, int, int]],
    rand_eff_dgps: list[EffectDist],
    B: int,
    C: int,
    n_repetitions: int,
) -> None:
    results_folder = Path(__file__).resolve().parent / "results"
    results_folder.mkdir(parents=True, exist_ok=True)

    # Write the experiment config to JSON
    config = {
        "base_seed": BASE_SEED,
        "sizes": sizes,
        "random_effect_dgps": rand_eff_dgps,
        "B": B,
        "C": C,
        "n_cpus": N_CPUS,
        **DGP_CONFIG,
        "confidence_levels": LEVELS,
        "n_repetitions": n_repetitions,
    }

    # Write the experiment config
    with open(results_folder / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    output_path = results_folder / "results.csv"

    if output_path.exists():
        output_path.unlink()

    n_scenarios = len(sizes) * len(rand_eff_dgps)

    # Master seed sequence used for deriving all the other seed sequences
    master_ss = np.random.SeedSequence(BASE_SEED)

    # For all scenarios, generate a seed sequence
    scenario_ss = master_ss.spawn(n_scenarios)

    scenario_idx = 0

    for rand_eff_dgp in rand_eff_dgps:
        print(f"Random effect DGP: {rand_eff_dgp}")
        for n_l3, n_l2, n_l1 in sizes:
            print(f"\tSample size: {n_l3}x{n_l2}x{n_l1}")
            # For each schenario, generate R seed sequences
            ss = scenario_ss[scenario_idx].spawn(n_repetitions)

            for i in range(n_repetitions):
                ss_data, ss_boot = ss[i].spawn(2)

                # Derive the data and bootstrap seed
                data_seed = ss_data.generate_state(1, dtype=np.uint64).item()

                # R seed has to be limited
                boot_seed = ss_boot.generate_state(
                    1, dtype=np.uint32
                ).item() % (2**31 - 1)

                data_sample = generate_hierarchical_dataset(
                    n_l3=n_l3,
                    n_l2=n_l2,
                    n_l1=n_l1,
                    **DGP_CONFIG,
                    random_state=data_seed,
                    random_eff_dist=rand_eff_dgp,
                )

                # Profile likelihood confidence intervals
                result_profile = compute_profile_likelihood_cis(
                    data_sample,
                    n_cpus=N_CPUS,
                )

                # Parametric percentile confidence intervals
                result_parametric_percentile = compute_parametric_percentile_ci(
                    data_sample,
                    B=B,
                    n_cpus=N_CPUS,
                    seed=boot_seed,
                )

                # Cases bootstrap (percentile and double)
                result_cases_perc, result_cases_double = (
                    compute_nonparametric_cases_ci(data_sample, B, C, boot_seed)
                )

                # Concatenate the results together
                simulation_result = pd.concat(
                    [
                        result_profile,
                        result_parametric_percentile,
                        result_cases_perc,
                        result_cases_double,
                    ],
                    ignore_index=True,
                )

                # Store extra metadata
                simulation_result["data_seed"] = data_seed
                simulation_result["boot_seed"] = boot_seed
                simulation_result["n_l3"] = n_l3
                simulation_result["n_l2"] = n_l2
                simulation_result["n_l1"] = n_l1
                simulation_result["rand_eff_dgp"] = rand_eff_dgp
                simulation_result["replication_id"] = i

                # Write to results file
                simulation_result.to_csv(
                    output_path,
                    header=not output_path.exists(),
                    mode="a",
                    index=False,
                )
            # Go to next scenario
            scenario_idx += 1


if __name__ == "__main__":
    # Experiment size regimes
    sizes = [
        (5, 4, 8),
        (50, 4, 8),
    ]

    rand_eff_dpgs: list[EffectDist] = ["norm", "t", "lognorm"]

    # Repeat each experiment configuration 1000 times
    n_repetitions = 1000

    # Number of resamples at the top level of the bootstrap
    B = 1000
    # Number of resamples at the second level of the bootstrap
    C = 20

    run_paired_experiment(
        sizes=sizes,
        rand_eff_dgps=rand_eff_dpgs,
        B=B,
        C=C,
        n_repetitions=n_repetitions,
    )

from pathlib import Path

# Reference library
import bootstrap_ci as boot
import numpy as np
import pandas as pd
from generators import DGP, DGPBiNorm, DGPExp, DGPLogNorm, DGPNorm
from numba import njit

# Our library
from bootstrap_diagnostics import Bootstrap, IIDResampler

# Base seed for seed sequence for repetitions
BASE_SEED = 42


# Seed the independent Numba RNG used by bootstrap_ci's JIT
# implementation of the nested bootstrap. Numba should also generate the same
# outer resamples as the percentile version (setting the same seed in Numba has
# the same behaviour as setting the same seed in NumPy) -> important since
# double bootstrap will resample the same indices for the outer resamples
# as the percentile one (bootstrap_ci only).
@njit
def seed_numba(seed):
    np.random.seed(seed)


# NOTE: bootstrap might fail to produce valid CIs for pearson correlation for
# small sample sizes (n=8 in our case). This function mimics the behaviour of
# np.corrcoef (but doesn't return the whole correlation matrix)
def batched_corr(data, axis):
    x = data[..., 0]
    y = data[..., 1]

    x = x - x.mean(axis, keepdims=True)
    y = y - y.mean(axis, keepdims=True)

    numerator = np.sum(x * y, axis)
    denominator = np.sqrt(np.sum(x**2, axis) * np.sum(y**2, axis))

    # Avoid errors but keep behaviour of corrcoef
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def corr(data):
    return np.corrcoef(data, rowvar=False)[0, 1]


def make_simulation_results(method, lower, upper, seeds):
    # We have the same replication_ids for the same samples for both libraries
    df = pd.DataFrame(
        {
            "method": method,
            # We need this to match libraries between resampled datasets for each experiment
            "replication_id": np.arange(len(lower)),
            "ci_lower": lower,
            "ci_upper": upper,
            # The seed used when constructing bootstrap CIs
            "bootstrap_seed": seeds,
        }
    )

    return df


def run_bootstrap_cis_our(samples, statistic, B, seeds):
    repetitions = len(samples)

    pb_upper_endpoints = np.empty(repetitions)
    pb_lower_endpoints = np.empty(repetitions)

    db_upper_endpoints = np.empty(repetitions)
    db_lower_endpoints = np.empty(repetitions)

    for i, (sample, seed) in enumerate(zip(samples, seeds, strict=True)):
        resampler = IIDResampler(sample)
        boot = Bootstrap(statistic, resampler, vectorized=True)

        pb_ci = boot.percentile_ci(
            0.95,
            side="two",
            b_resamples=B,
            n_jobs=-1,
            seed=seed,
            q_est_method="median_unbiased",
        )

        db_ci = boot.double_percentile_ci(
            0.95,
            side="two",
            b1_resamples=B,
            b2_resamples=B,
            n_jobs=-1,
            seed=seed,
            q_est_method="median_unbiased",
        )

        pb_upper_endpoints[i] = pb_ci.upper
        pb_lower_endpoints[i] = pb_ci.lower
        db_upper_endpoints[i] = db_ci.upper
        db_lower_endpoints[i] = db_ci.lower

    results = [
        make_simulation_results(
            method="percentile",
            lower=pb_lower_endpoints,
            upper=pb_upper_endpoints,
            seeds=seeds,
        ),
        make_simulation_results(
            method="double",
            lower=db_lower_endpoints,
            upper=db_upper_endpoints,
            seeds=seeds,
        ),
    ]

    # Form a DataFrame from both data frames by stacking them on top of each
    # other
    return pd.concat(results, ignore_index=True)


def run_bootstrap_cis_ref(samples, statistic, B, seeds):
    repetitions = len(samples)

    pb_upper_endpoints = np.empty(repetitions)
    pb_lower_endpoints = np.empty(repetitions)

    db_upper_endpoints = np.empty(repetitions)
    db_lower_endpoints = np.empty(repetitions)

    for i, (sample, seed) in enumerate(zip(samples, seeds, strict=True)):
        bs = boot.Bootstrap(sample, statistic, use_jit=True)

        pb_ci = bs.ci(
            coverages=[0.025, 0.975],
            side="one",
            method="percentile",
            nr_bootstrap_samples=B,
            seed=seed,
            quantile_type="median_unbiased",
        )

        # Recreate the Bootstrap object so we evaluate the statistic again on
        # the outside resamples - prevent using cached percentile state
        bs = boot.Bootstrap(sample, statistic, use_jit=True)

        seed_numba(seed)
        db_ci = bs.ci(
            coverages=[0.025, 0.975],
            side="one",
            method="double",
            nr_bootstrap_samples=B,
            seed=seed,
            quantile_type="median_unbiased",
        )

        pb_upper_endpoints[i] = pb_ci[1]
        pb_lower_endpoints[i] = pb_ci[0]

        db_upper_endpoints[i] = db_ci[1]
        db_lower_endpoints[i] = db_ci[0]

    results = [
        make_simulation_results(
            method="percentile",
            lower=pb_lower_endpoints,
            upper=pb_upper_endpoints,
            seeds=seeds,
        ),
        make_simulation_results(
            method="double",
            lower=db_lower_endpoints,
            upper=db_upper_endpoints,
            seeds=seeds,
        ),
    ]

    return pd.concat(results, ignore_index=True)


def run_paired_experiment(dgps, statistics, ns, B, repetitions):
    # Join results from both libraries into a single CSV
    results_folder = Path(__file__).resolve().parent / "results"

    results_folder.mkdir(parents=True, exist_ok=True)

    output_path = results_folder / "results.csv"

    if output_path.exists():
        output_path.unlink()

    # Use seed sequences for better reproducibility
    ss = np.random.SeedSequence(BASE_SEED)
    replication_sequences = ss.spawn(repetitions)
    seeds = [
        s.generate_state(1, dtype=np.uint32).item()
        for s in replication_sequences
    ]

    for dgp in dgps:
        for stat_name, (stat_our, stat_ref) in statistics.items():
            is_corr = stat_name == "corr"

            # We only use correlation and bivariate normal in combination with each other
            if isinstance(dgp, DGPBiNorm) != is_corr:
                continue

            for n in ns:
                print(
                    f"Currently evaluating: dgp={dgp.describe()}, statistic={stat_ref.__name__}, n={n}"
                )

                # Sample in batch so both libraries get the same samples
                samples = dgp.sample(sample_size=n, nr_samples=repetitions)
                # stat_ref is the safest to use since it doesn't include batch_corr
                true_value = dgp.get_true_value(stat_name)

                results_our = run_bootstrap_cis_our(samples, stat_our, B, seeds)
                results_ref = run_bootstrap_cis_ref(samples, stat_ref, B, seeds)

                # Mark the results with library indices
                results_our["library"] = "our"
                results_ref["library"] = "ref"

                df_joined = pd.concat(
                    [results_our, results_ref], ignore_index=True
                )

                assert len(df_joined) == 4 * repetitions, (
                    f"Expected data frame of length {4 * repetitions}; got {len(df_joined)}"
                )

                df_joined["dgp"] = dgp.describe()
                df_joined["statistic"] = stat_name
                df_joined["n"] = n
                df_joined["B"] = B
                df_joined["true_param_value"] = true_value
                df_joined["nominal_1_sided_coverage"] = 0.975
                df_joined["nominal_2_sided_coverage"] = 0.95

                df_joined.to_csv(
                    output_path,
                    header=not output_path.exists(),
                    mode="a",
                    index=False,
                )


# Note that in this validation we use 0.025 and 0.975 confidence bounds for one
# sided intervals which are also used for construction of two sided intervals
# with 0.95 nominal coverage
if __name__ == "__main__":
    seed = 0

    # Sample sizes
    ns = [8, 32, 128]

    # Repeat each each experiment 1000 times
    repetitions = 1000

    # Use B(=C)=1000 resamples for the bootstrap
    B = 1000

    dgps = [
        DGPNorm(seed, 0, 1),
        DGPExp(seed, 1),
        DGPBiNorm(seed, np.array([1, 1]), np.array([[2, 0.5], [0.5, 1]])),
        DGPLogNorm(seed, 0, 1),
    ]

    # statistic_name -> our_stat, ref_stat
    statistics = {
        "mean": (np.mean, np.mean),
        "median": (np.median, np.median),
        "corr": (batched_corr, corr),
    }

    # Run the whole experiment together so we guarantee the same external samples
    run_paired_experiment(
        dgps,
        statistics,
        ns,
        B,
        repetitions,
    )

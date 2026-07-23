from pathlib import Path

# Reference library
import bootstrap_ci as boot
import numpy as np
import pandas as pd
from generators import DGP, DGPBiNorm, DGPExp, DGPLogNorm, DGPNorm

# Our library
from bootstrap_diagnostics import Bootstrap, IIDResampler


# Batched version of Pearson correlation so vectorization can be used
def batched_corr(data, axis):
    x = data[..., 0]
    y = data[..., 1]

    x = x - x.mean(axis, keepdims=True)
    y = y - y.mean(axis, keepdims=True)

    numerator = np.sum(x * y, axis)
    denominator = np.sqrt(np.sum(x**2, axis) * np.sum(y**2, axis))

    # Avoid errors for degenerate samples
    return np.divide(
        numerator,
        denominator,
        np.zeros_like(numerator),
        dtype=float,
        where=denominator != 0,
    )


def corr(data):
    return np.corrcoef(data, rowvar=False)[0, 1]


def aggregate_simulation_results(method, true_value, lower, upper):
    return {
        "method": method,
        "nominal_1_sided_coverage": 0.975,
        "nominal_2_sided_coverage": 0.95,
        "lower_coverage": np.mean(lower <= true_value),
        "upper_coverage": np.mean(true_value <= upper),
        "two_sided_coverage": np.mean(
            (lower <= true_value) & (true_value <= upper)
        ),
        "mean_length": np.mean(upper - lower),
        "median_length": np.median(upper - lower),
        "valid_interval_rate": np.mean(
            np.isfinite(upper) & np.isfinite(lower) & (lower <= upper)
        ),
    }


def run_bootstrap_cis_our(dgp, statistic, B, n, repetitions):
    pb_upper_endpoints = np.empty(repetitions)
    pb_lower_endpoints = np.empty(repetitions)

    db_upper_endpoints = np.empty(repetitions)
    db_lower_endpoints = np.empty(repetitions)

    samples = dgp.sample(nr_samples=repetitions, sample_size=n)

    for i, sample in enumerate(samples):
        resampler = IIDResampler(sample)
        boot = Bootstrap(statistic, resampler, vectorized=True)

        pb_ci = boot.percentile_ci(
            0.95,
            side="two",
            b_resamples=B,
            n_jobs=-1,
            seed=i,
        )

        db_ci = boot.double_percentile_ci(
            0.95,
            side="two",
            b1_resamples=B,
            b2_resamples=B,
            n_jobs=-1,
            seed=i,
        )

        pb_upper_endpoints[i] = pb_ci.upper
        pb_lower_endpoints[i] = pb_ci.lower
        db_upper_endpoints[i] = db_ci.upper
        db_lower_endpoints[i] = db_ci.lower

    # We have to be careful with batched_corr
    statistic_name = statistic.__name__
    true_value_name = (
        "corr" if statistic_name == "batched_corr" else statistic_name
    )
    true_value = dgp.get_true_value(true_value_name)

    results = []

    for method, lower, upper in [
        ("percentile", pb_lower_endpoints, pb_upper_endpoints),
        ("double", db_lower_endpoints, db_upper_endpoints),
    ]:
        results.append(
            aggregate_simulation_results(
                method,
                true_value,
                lower,
                upper,
            )
        )

    return results


def run_bootstrap_cis_ref(dgp, statistic, B, n, repetitions):
    pb_upper_endpoints = np.empty(repetitions)
    pb_lower_endpoints = np.empty(repetitions)

    db_upper_endpoints = np.empty(repetitions)
    db_lower_endpoints = np.empty(repetitions)

    samples = dgp.sample(nr_samples=repetitions, sample_size=n)

    for i, sample in enumerate(samples):
        # Use just in time compilation to make it faster
        bs = boot.Bootstrap(sample, statistic, use_jit=True)

        pb_ci = bs.ci(
            coverages=[0.025, 0.975],
            side="one",
            method="percentile",
            nr_bootstrap_samples=B,
            seed=i,
        )

        db_ci = bs.ci(
            coverages=[0.025, 0.975],
            side="one",
            method="double",
            nr_bootstrap_samples=B,
            seed=i,
        )

        pb_upper_endpoints[i] = pb_ci[1]
        pb_lower_endpoints[i] = pb_ci[0]

        db_upper_endpoints[i] = db_ci[1]
        db_lower_endpoints[i] = db_ci[0]

    true_value = dgp.get_true_value(statistic.__name__)

    results = []

    for method, lower, upper in [
        ("percentile", pb_lower_endpoints, pb_upper_endpoints),
        ("double", db_lower_endpoints, db_upper_endpoints),
    ]:
        results.append(
            aggregate_simulation_results(
                method,
                true_value,
                lower,
                upper,
            )
        )

    return results


def run(dgps, ns, statistics, B, repetitions, our=True):
    if our:
        results_folder = Path(__file__).resolve().parent / "results_our"
        output_path = results_folder / "results_our.csv"
    else:
        results_folder = Path(__file__).resolve().parent / "results_ref"
        output_path = results_folder / "results_ref.csv"

    results_folder.mkdir(parents=True, exist_ok=True)

    # Delete existing file
    if output_path.exists():
        output_path.unlink()

    for dgp in dgps:
        for statistic in statistics:
            # Special case - only evaluate correlation for bivariate normal
            is_corr = statistic.__name__ in ("corr", "batched_corr")
            if is_corr != isinstance(dgp, DGPBiNorm):
                continue

            for n in ns:
                print(
                    f"Currently doing: dgp={dgp.describe()}, statistic={statistic.__name__ if not is_corr else 'corr'}, n={n}"
                )

                if our:
                    result = run_bootstrap_cis_our(
                        dgp,
                        statistic,
                        B,
                        n,
                        repetitions,
                    )
                else:
                    result = run_bootstrap_cis_ref(
                        dgp,
                        statistic,
                        B,
                        n,
                        repetitions,
                    )

                result_df = pd.DataFrame(result)

                result_df["n"] = n
                result_df["dgp"] = dgp.describe()
                result_df["statistic"] = (
                    statistic.__name__
                    if statistic.__name__ != "batched_corr"
                    else "corr"
                )
                result_df["B"] = B
                result_df["repetitions"] = repetitions
                result_df.to_csv(
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
    repetitions = 100

    # Use B(=C)=1000 resamples for the bootstrap
    B = 100

    dgps_our = [
        DGPNorm(seed, 0, 1),
        DGPExp(seed, 1),
        DGPBiNorm(seed, np.array([1, 1]), np.array([[2, 0.5], [0.5, 1]])),
        DGPLogNorm(seed, 0, 1),
    ]

    dgps_new = [
        DGPNorm(seed, 0, 1),
        DGPExp(seed, 1),
        DGPBiNorm(seed, np.array([1, 1]), np.array([[2, 0.5], [0.5, 1]])),
        DGPLogNorm(seed, 0, 1),
    ]

    # Statistics for our library (we use batched correlation since we support vectorization)
    statistics_our = [np.mean, np.median, batched_corr]

    # Statistics for the reference library (use non-batched correlation)
    statistics_ref = [
        np.mean,
        np.median,
        corr,
    ]

    # Get results for our library
    print("Running our library")
    run(dgps_our, ns, statistics_our, B, repetitions, our=True)

    # Get results for the reference library
    print("Running reference library")
    run(dgps_new, ns, statistics_ref, B, repetitions, our=False)

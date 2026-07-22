from pathlib import Path

import numpy as np
import pandas as pd
from generators import DGP, DGPBiNorm, DGPExp, DGPNorm

from bootstrap_diagnostics import Bootstrap, IIDResampler


# Batched version of corrcoef so vectorization can be used
def corr(data, axis):
    x = data[..., 0]
    y = data[..., 1]

    x = x - x.mean(axis, keepdims=True)
    y = y - y.mean(axis, keepdims=True)

    numerator = np.sum(x * y, axis)
    denominator = np.sqrt(np.sum(x**2, axis) * np.sum(y**2, axis))

    return numerator / denominator


def run_bootstrap_cis(dgp, statistic, B, n, repetitions):
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

    true_value = dgp.get_true_value(statistic.__name__)

    results = []

    for method, lower, upper in [
        ("percentile", pb_lower_endpoints, pb_upper_endpoints),
        ("double", db_lower_endpoints, db_upper_endpoints),
    ]:
        results.append(
            {
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
        )

    return results


def run_comparison(dgps, ns, statistics, B, repetitions):
    results_folder = Path(__file__).resolve().parent / "results_eh_es"
    results_folder.mkdir(parents=True, exist_ok=True)

    output_path = results_folder / "results_eh_es.csv"

    if output_path.exists():
        output_path.unlink()

    for dgp in dgps:
        for statistic in statistics:
            if (
                statistic.__name__ == "corr" and not isinstance(dgp, DGPBiNorm)
            ) or (isinstance(dgp, DGPBiNorm) and statistic.__name__ != "corr"):
                continue

            for n in ns:
                print(
                    f"Currently doing: dgp={dgp.describe()}, statistic={statistic.__name__}, n={n}"
                )
                result = run_bootstrap_cis(dgp, statistic, B, n, repetitions)
                result_df = pd.DataFrame(result)

                result_df["n"] = n
                result_df["dgp"] = dgp.describe()
                result_df["statistic"] = statistic.__name__
                result_df["B"] = B
                result_df["repetitions"] = repetitions
                result_df.to_csv(
                    output_path,
                    header=not output_path.exists(),
                    mode="a",
                    index=False,
                )


if __name__ == "__main__":
    seed = 0
    ns = [8, 32, 128]
    repetitions = 1000
    B = 1000

    dgps = [
        DGPNorm(seed, 0, 1),
        DGPExp(seed, 1),
        DGPBiNorm(seed, np.array([1, 1]), np.array([[2, 0.5], [0.5, 1]])),
    ]

    statistics = [np.mean, np.median, corr]

    run_comparison(dgps, ns, statistics, B, repetitions)

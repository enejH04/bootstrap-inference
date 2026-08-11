from typing import Literal

import numpy as np
import pandas as pd

EffectDist = Literal["norm", "t", "lognorm"]


def generate_hierarchical_dataset(
    n_l3: int,
    n_l2: int,
    n_l1: int,
    mu: float = 0,
    var_l3: float = 0.3,
    var_l2: float = 0.3,
    var_l1: float = 0.4,
    random_state: np.random.SeedSequence | int | None = None,
    random_eff_dist: EffectDist = "norm",
) -> pd.DataFrame:
    # For example l3: schools, l2: classrooms, l1: student scores on an exam
    # Total number of observations in the dataset
    n_observations = n_l3 * n_l2 * n_l1

    # We store 3 columns: level3 id, level2 id, and response
    data = np.empty(shape=(n_observations, 3))

    # Get the rng based on the provided seed sequence
    rng = np.random.default_rng(random_state)

    # Starting index of rows that we are filling up
    row = 0

    for i in range(n_l3):
        l3_eff = draw_effect(rng, random_eff_dist) * np.sqrt(var_l3)

        for j in range(n_l2):
            l2_eff = draw_effect(rng, random_eff_dist) * np.sqrt(var_l2)

            # Residuals are always normally distributed
            residuals = rng.standard_normal(size=n_l1) * np.sqrt(var_l1)

            # Compute the result
            results = mu + l3_eff + l2_eff + residuals

            # Set the identifiers and response
            data[row : row + n_l1, 0] = i
            data[row : row + n_l1, 1] = j
            data[row : row + n_l1, 2] = results

            row += n_l1

    df = pd.DataFrame(data, columns=["l3", "l2", "y"]).astype(
        {"l3": "int64", "l2": "int64", "y": "float64"}
    )

    return df


def draw_effect(
    rng: np.random.Generator,
    random_eff_dist: EffectDist,
) -> float:
    #  Sample random effects from standardised distributions
    match random_eff_dist:
        case "norm":
            eff = rng.standard_normal()
        case "t":
            eff = rng.standard_t(df=3) / np.sqrt(3)
        case "lognorm":
            e = np.exp(1)

            eff = ((rng.lognormal(mean=0, sigma=1)) - np.sqrt(e)) / np.sqrt(
                e * (e - 1)
            )
        case _:
            raise ValueError(
                f"Unknown random effect distribution: {random_eff_dist}"
            )

    return eff


if __name__ == "__main__":
    # Test to see if balanced hierarchy is preserved
    from bootstrap_diagnostics import HierarchicalResampler

    n_l3 = 5
    n_l2 = 4
    n_l1 = 8

    ss = np.random.SeedSequence(123)
    df = generate_hierarchical_dataset(n_l3, n_l2, n_l1, random_state=ss)

    resampler = HierarchicalResampler(
        data_sample=df,
        hierarchy=[("l3", True), ("l2", True)],
        observation_replacement=True,
    )

    sample = resampler.draw_sample(np.random.default_rng(42))

    assert len(sample) == n_l3 * n_l2 * n_l1
    assert sample["l3"].nunique() == n_l3
    assert (sample.groupby("l3")["l2"].nunique() == n_l2).all()
    assert (sample.groupby(["l3", "l2"]).size() == n_l1).all()

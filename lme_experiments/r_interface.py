import os
import sysconfig
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# This fixes some environment issues with running the R code in multiple
# processes using joblib, where workers load the wrong Python due to rpy2
# modifying the environment
python_libdir = sysconfig.get_config_var("LIBDIR")

if python_libdir:
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    paths = [python_libdir, *(p for p in existing if p and p != python_libdir)]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)


# Function definitions in R
R_DEFS = Path(__file__).resolve().parent / "lme4_helpers.R"

# Load the R file with helpers
ro.r["source"](str(R_DEFS))  # ty:ignore[call-non-callable]

# Get the statistic defined in R
stat = ro.globalenv["statistic"]
profile_cis = ro.globalenv["profile_cis"]
parametric_boot_cis = ro.globalenv["parametric_boot_cis"]

# Convert between Pandas DataFrame and R data.frame objects
converter = ro.default_converter + pandas2ri.converter

CI_COLUMNS = ["0.025", "0.05", "0.25", "0.75", "0.95", "0.975"]
STATS = ["mu", "sd_l3", "sd_l2"]


def pd_to_r(df: pd.DataFrame):
    with converter.context():
        r_df = ro.conversion.get_conversion().py2rpy(df)

    return r_df


def construct_ci_df(ci_output: npt.NDArray, method_name: str) -> pd.DataFrame:
    df = pd.DataFrame(
        ci_output,
        columns=[*CI_COLUMNS, "estimate"],
    )
    df["stat"] = STATS
    df["method"] = method_name

    return df


# Custom statistic for the cases bootstrap
def statistic(df: pd.DataFrame) -> npt.NDArray:
    # Convert to R data.frame
    r_df = pd_to_r(df)

    # Fit the lme and get the values for parameters of interest
    result = stat(r_df)

    # Making a copy is important here!
    return np.array(result, dtype=np.float64, copy=True)


def compute_profile_likelihood_cis(
    df: pd.DataFrame,
    n_cpus: int = 1,
) -> pd.DataFrame:
    r_df = pd_to_r(df)

    result = np.array(
        profile_cis(r_df, n_cpus=n_cpus),
    )

    return construct_ci_df(result, "profile-likelihood")


def compute_parametric_percentile_ci(
    df: pd.DataFrame,
    B: int = 1000,
    n_cpus: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    r_df = pd_to_r(df)

    result = np.array(
        parametric_boot_cis(r_df, B=B, seed=seed, n_cpus=n_cpus),
    )

    return construct_ci_df(result, "parametric-percentile-boot")

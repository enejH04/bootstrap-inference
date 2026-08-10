from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# Function definitions in R
R_DEFS = Path(__file__).parent / "lme4_helpers.R"

# Load the R file with helpers
ro.r["source"](str(R_DEFS))  # ty:ignore[call-non-callable]

# Get the statistic defined in R
stat = ro.globalenv["statistic"]
profile_cis = ro.globalenv["profile_cis"]

# Convert between Pandas DataFrame and R data.frame objects
converter = ro.default_converter + pandas2ri.converter


# Custom statistic for the cases bootstrap
def statistic(df: pd.DataFrame) -> npt.NDArray:
    # Convert to R data.frame
    with converter.context():
        r_df = ro.conversion.get_conversion().py2rpy(df)

    # Fit the lme and get the values for parameters of interest
    result = stat(r_df)

    # Making a copy is important here!
    return np.array(result, dtype=np.float64, copy=True)


def compute_profile_cis(
    df: pd.DataFrame,
    levels: list[float] = [0.50, 0.90, 0.975],
) -> pd.DataFrame:
    with converter.context():
        r_df = ro.conversion.get_conversion().py2rpy(df)

    result = np.array(
        profile_cis(r_df, levels=levels),
    )

    df = pd.DataFrame(
        result, columns=["0.025", "0.05", "0.25", "0.75", "0.95", "0.975"]
    )
    df["stat"] = ["mu", "sd_l3", "sd_l2"]

    return df

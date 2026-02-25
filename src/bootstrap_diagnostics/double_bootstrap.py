from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt


@dataclass(frozen=True)
class ConfidenceInterval:
    """Result of the bootstrap confidence interval procedure."""

    confidence_level: float
    side: Literal["two", "left", "right"]
    lower: float
    upper: float

    def __str__(self) -> str:
        return f"CI = ({self.lower}, {self.upper})"

    # TODO: diagnostic results could potentially be included here?


class DoubleBootstrap:
    """
    Nonparametric bootstrap class for scalar statistics.

    Implements nonparametric double percentile bootstrap procedure for
    confidence interval construction.


    Parameters
    ----------
    data_sample : npt.ArrayLike
        A dataset that can be converted to a NumPy array of floats.
        For multivariate data, the array must have shape ``(n, d)``, where ``n``
        is the number of observations and ``d`` the dimension of the observed
        data points.

    statistic : Callable[[npt.ArrayLike], float]
        The function used to calculate the statistic of interest.
        Must follow the signature ``f(data) -> float``.

    axis : int, optional
        The axis along which to compute the statistic. Defaults to 0.

    Raises
    ------
    ValueError
        If ``data_sample`` cannot be converted to a Numpy float array or is empty.
    """

    # TODO: handle multi-dimensional data better (specify axis so user has more control)
    def __init__(
        self,
        data_sample: npt.ArrayLike,
        statistic: Callable[[npt.ArrayLike], float],
        axis: int = 0,
    ) -> None:
        try:
            self._data_sample = np.asarray(data_sample, dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(
                "Cannot convert given array of data points to a Numpy float array"
            )
        if len(self._data_sample) == 0:
            raise ValueError(
                "Input data sample is empty. Cannot perform bootstrap"
            )

        # # Convert a one-dimensional (row) vector of shape (n,) to (n, 1) to
        # # unify the implementation to multivariate data
        # if self._data_sample.ndim == 1:
        #     self._data_sample = self._data_sample.reshape(-1, 1)

        self._statistic = statistic
        self._axis = axis

    def confidence_interval(
        self,
        confidence_level: float = 0.95,
        side: Literal["two", "left", "right"] = "two",
        B1_resamples: int = 1000,
        B2_resamples: int = 250,
        q_est_method: str = "median_unbiased",
        seed: Optional[int] = None,
    ) -> ConfidenceInterval:
        """
        Compute a confidence interval of the sample data.

        Parameters
        ----------
        confidence_level : float, optional
            The confidence level of the interval. Defaults to 0.95.
        side : {"two", "lower", "upper"}, optional
            The sideness of the interval. "two" for two-sided and "lower",
            "upper" for one-sided. Defaults to "two".
        B1_resamples : int, optional
            Number of bootstrap resamples in the first level. Defaults to 1000.
        B2_resamples : int, optional
            Number of bootstrap resamples in the second level for calibration
            of the percentile method. Defaults to 250.
        q_est_method: str, optional
            Method for quantile estimation. Passed as ``numpy.quantile``'s argument ``method``.
            Defaults to "median_unbiased".
        seed : int, optional
            Seed for the random number generator. Defaults to None.

        Returns
        -------
        ConfidenceInterval
            The computed bootstrap confidence interval.

        Raises
        ------
        ValueError
            If ``confidence_level`` is not in (0, 1), ``side`` is invalid
            or ``n_resamples <= 0``.
        """

        # Use this function as a wrapper to make adding new methods easier and
        # decoupled from the validation logic
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"Confidence_level should be (0, 1); got {confidence_level}"
            )
        if side not in {"two", "left", "right"}:
            raise ValueError(
                f"Side must be 'two', 'left' or 'right'; got {side}"
            )
        if B1_resamples <= 0 or B2_resamples <= 0:
            raise ValueError("Number of resamples must be positive")

        rng = np.random.default_rng(seed)

        return self._double_percentile_ci(
            confidence_level,
            side,
            B1_resamples,
            B2_resamples,
            q_est_method,
            rng,
        )

    # TODO: speed this up - this is just a naive implementation currently to test
    # performance
    def _double_percentile_ci(
        self,
        confidence_level: float,
        side: Literal["two", "left", "right"],
        B1: int,
        B2: int,
        q_est_method: str,
        rng: Generator,
    ) -> ConfidenceInterval:
        """
        Internal method that computes the CI using the double percentile
        bootstrap method.

        Parameters
        ----------
        confidence_level: float
            The confidence level of the interval.
        side : {"two", "lower", "upper"}
            The sideness of the interval. "two" for two-sided and "lower",
            "upper" for one-sided.
        B1 : int, optional
            Number of bootstrap resamples in the first level.
        B2 : int, optional
            Number of bootstrap resamples in the second level for calibration
            of the percentile method.
        q_est_method: str
            Method for quantile estimation. Passed as ``numpy.quantile``'s argument ``method``.
        rng : Generator
            Numpy random number generator.

        Returns
        -------
        ConfidenceInterval
            The computed bootstrap confidence interval.
        """

        # Evaluate the statistic on the original sample (needed for calibration)
        estimate = self._statistic(self._data_sample)

        # First level matrix of dataset indices
        b1_matrix = self._resample_indices(
            len(self._data_sample),
            B1,
            rng,
        )
        # Level 1 estimates
        # TODO: this could be cleaner
        l1_estimates: list[float] = []
        cdf_evals: list[float] = []

        for b1_indices in b1_matrix:
            # Use np.take to index along the specified axis to unify the implementation for multi-dimensional data
            b1_data = np.take(self._data_sample, b1_indices, axis=self._axis)

            # Store statistic evaluated on the bootstrapped dataset
            l1_estimates.append(self._statistic(b1_data))

            # Second level matrix of dataset indices corresponding to instances in b1_data
            b2_matrix = self._resample_indices(
                len(b1_data),
                B2,
                rng,
            )

            # Compute the level 2 estimates
            l2_estimates = np.array(
                [
                    self._statistic(
                        np.take(b1_data, b2_indices, axis=self._axis)
                    )
                    for b2_indices in b2_matrix
                ]
            )
            eval = np.mean(l2_estimates <= estimate)
            cdf_evals.append(eval)

        # Quantile estimation method
        alpha = 1 - confidence_level

        # Adjust values to get more accurate coverage
        match side:
            case "two":
                alpha_low_DB, alpha_high_db = np.quantile(
                    cdf_evals,
                    [alpha / 2, 1 - alpha / 2],
                    method=q_est_method,
                )
                lower, upper = np.quantile(
                    l1_estimates,
                    [alpha_low_DB, alpha_high_db],
                    method=q_est_method,
                )
            case "upper":
                alpha_DB = np.quantile(
                    cdf_evals,
                    1 - alpha,
                    method=q_est_method,
                )
                lower, upper = (
                    -np.inf,
                    np.quantile(
                        l1_estimates,
                        alpha_DB,
                        method=q_est_method,
                    ),
                )
            case "lower":
                alpha_DB = np.quantile(
                    cdf_evals,
                    alpha,
                    method=q_est_method,
                )
                lower, upper = (
                    np.quantile(l1_estimates, alpha_DB, method=q_est_method),
                    np.inf,
                )

        return ConfidenceInterval(
            confidence_level,
            side,
            lower,
            upper,
        )

    def _resample_indices(
        self,
        n_instances: int,
        n_resamples: int,
        rng: Generator,
    ) -> npt.NDArray[np.intp]:
        """
        Resample B bootstrapped datasets (indirectly through indices).
        The instances are sampled with replacement.

        Parameters
        ----------
        n_instances : int
            Number of instances in the dataset.
        n_resamples : int
            Number of bootstrap resampled datasets to generate.
        rng : Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray[np.intp]
            A matrix of shape ``(n_resamples, n_instances)``, where each row
            represents is an array of indices that correspond to instances.
        """
        resampled_datasets = rng.integers(
            0, n_instances, size=(n_resamples, n_instances), dtype=np.intp
        )

        return resampled_datasets

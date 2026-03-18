from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
from numpy.random import Generator
import numpy.typing as npt

from joblib import Parallel, delayed


@dataclass(frozen=True)
class ConfidenceInterval:
    """Result of the bootstrap confidence interval procedure."""

    confidence_level: float
    side: Literal["two", "lower", "upper"]

    # Allow both scalar and array confidence intervals
    # (e.g. regression coefficients of multiple linear regression)
    lower: npt.NDArray[np.float64] | float
    upper: npt.NDArray[np.float64] | float

    def __str__(self) -> str:
        return f"lower = {self.lower}, upper = {self.upper}, confidence level = {self.confidence_level}, side = {self.side}"

    # TODO: diagnostic results could potentially be included here?


class DoubleBootstrap:
    """
    Nonparametric bootstrap class for scalar statistics.

    Implements nonparametric double percentile bootstrap procedure for
    confidence interval construction.

    Note that the CIs for non-scalar statistics are component-wise and do not
    represent a joint confidence region.

    Parameters
    ----------
    data_sample : npt.ArrayLike
        A dataset that can be converted to a NumPy array of floats.
        For multivariate data, the statistical functional will be computed along the specified ``axis``.

    statistic : Callable[[npt.ArrayLike], npt.NDArray[np.float64]]
        The function used to calculate the statistic of interest.
        Must follow the signature ``f(data) -> npt.NDArray[np.float64]``.

    axis : int, optional
        The axis along which to compute the statistic. Defaults to 0.

    Raises
    ------
    ValueError
        If ``data_sample`` cannot be converted to a Numpy float array or is empty.
    """

    def __init__(
        self,
        data_sample: npt.ArrayLike,
        statistic: Callable[[npt.ArrayLike], npt.NDArray[np.float64] | float],
        axis: int = 0,
    ) -> None:
        try:
            # Try to convert the input data sample to a Numpy array of floats
            # TODO: add a wrapper for pandas data frames
            self._data_sample = np.asarray(data_sample, dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(
                "Cannot convert given array of data points to a Numpy float array"
            )
        if len(self._data_sample) == 0:
            raise ValueError(
                "Input data sample is empty. Cannot perform bootstrap"
            )

        self._statistic = statistic
        self._axis = axis

    def confidence_interval(
        self,
        confidence_level: float = 0.95,
        side: Literal["two", "lower", "upper"] = "two",
        B1_resamples: int = 1000,
        B2_resamples: int = 250,
        q_est_method: str = "median_unbiased",
        n_jobs: int = 1,
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
        n_jobs: int, optional
            Number of concurrent jobs used for the bootstrap procedure.
            Follows the Joblib convention: -1 tries to use all CPUs, 1 disables parallelism.
            Defaults to 1.
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
        if side not in {"two", "lower", "upper"}:
            raise ValueError(
                f"Side must be 'two', 'lower' or 'upper'; got {side}"
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
            n_jobs,
            rng,
        )

    def _double_percentile_ci(
        self,
        confidence_level: float,
        side: Literal["two", "lower", "upper"],
        B1: int,
        B2: int,
        q_est_method: str,
        n_jobs: int,
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
        n_jobs: int, optional
            Number of concurrent jobs used for the bootstrap procedure.
            Follows the Joblib convention: -1 tries to use all CPUs, 1 disables parallelism.
            Defaults to 1.
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
            self._data_sample.shape[self._axis],
            B1,
            rng,
        )

        # Derive the seeds for all B1 jobs using the provided RNG
        # TODO: look into whether this RNG is ok and reproducible
        # I believe this should be fine since we deterministically derive
        # the seed for each child from the original seed
        seeds = rng.integers(0, 2**32, size=B1)

        # Delegate the tasks
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_b1)(estimate, b1_matrix[i], B2, seeds[i])
            for i in range(B1)
        )

        # zip((1, 2), (3, 4)) = (1, 3), (2, 4)
        # Since we are working with potentially multi-dimensional statistics,
        # we need to store the estimates in an array of the same shape as the
        # statistic output, but with an additional dimension for the
        # number of resamples.
        # For example, if the statistic is a vector of length k, the shape of l1_estimates will be (B1, k).
        # (basically stack them on top of each other)
        l1_estimates, cdf_evals = zip(*results)
        l1_estimates = np.array(l1_estimates)
        cdf_evals = np.array(cdf_evals)

        # Quantile estimation method
        alpha = 1 - confidence_level

        # Adjust values to get more accurate coverage
        match side:
            case "two":
                alpha_lower_DB, alpha_upper_DB = np.quantile(
                    cdf_evals,
                    [alpha / 2, 1 - alpha / 2],
                    axis=0,
                    method=q_est_method,
                )
                lower = self._quantile_per_component(
                    l1_estimates, alpha_lower_DB, q_est_method
                )
                upper = self._quantile_per_component(
                    l1_estimates, alpha_upper_DB, q_est_method
                )
            case "upper":
                alpha_DB = np.quantile(
                    cdf_evals,
                    1 - alpha,
                    axis=0,
                    method=q_est_method,
                )
                lower = -np.inf * np.ones_like(estimate)
                upper = self._quantile_per_component(
                    l1_estimates, alpha_DB, q_est_method
                )
            case "lower":
                alpha_DB = np.quantile(
                    cdf_evals,
                    alpha,
                    axis=0,
                    method=q_est_method,
                )
                lower = self._quantile_per_component(
                    l1_estimates, alpha_DB, q_est_method
                )
                upper = np.inf * np.ones_like(estimate)

        return ConfidenceInterval(
            confidence_level,
            side,
            lower,
            upper,
        )

    def _process_b1(
        self,
        estimate: npt.NDArray[np.float64],
        b1_indices: npt.NDArray[np.intp],
        B2: int,
        seed: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Internal method that computes the double bootstrap procedure for a single
        first level resample.

        Parameters
        ----------
        estimate : npt.NDArray[np.float64]
            The estimate of the statistic computed from the original sample.
        b1_indices :
            First level vector of instance indices which represent our bootstrap
            resample.
        B2 : int
            Number of bootstrap resamples in the second level for calibration
            of the percentile method.
        seed : int
            Random seed used for resampling from the provided dataset.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            The level 1 bootstrap estimate computed from the resample and G^*(\hat{\theta})
        """

        # Instantiate a local RNG that is unique to this process
        local_rng = np.random.default_rng(seed)

        # Use np.take to index along the specified axis to unify the implementation for multi-dimensional data
        b1_data = np.take(self._data_sample, b1_indices, axis=self._axis)

        # Compute the level 1 bootstrap estimate
        l1_estimate = self._statistic(b1_data)

        # Second level matrix of dataset indices corresponding to instances in b1_data
        b2_matrix = self._resample_indices(
            b1_data.shape[self._axis],
            B2,
            local_rng,
        )

        # Compute the level 2 estimates
        l2_estimates = np.array(
            [
                self._statistic(np.take(b1_data, b2_indices, axis=self._axis))
                for b2_indices in b2_matrix
            ]
        )

        # How many level 2 estimates are less than or equal to the original estimate?
        # CDF evaluation G^*(\hat{\theta})
        # Compute the mean along the axis that correspond to the same component
        # of the statistic (e.g. if the statistic is a vector, compute the mean for each component separately)
        cdf_eval = np.mean(l2_estimates <= estimate, axis=0)

        return l1_estimate, cdf_eval

    def _quantile_per_component(
        self,
        data: npt.NDArray[np.float64],
        quantiles: npt.NDArray[np.float64],
        q_est_method: str,
    ):
        """
        data: npt.NDArray[np.float64]
            Data we want to compute per component quantiles for.
        quantiles: npt.NDArray[np.float64]
            Per component quantiles. Need to have the same shape as data[i].
        q_est_method: str
            Method for quantile estimation. Passed as ``numpy.quantile``'s argument ``method``.
        """
        # Create a (B1, n_components) vector
        flat = data.reshape(data.shape[0], -1)
        flat_a = quantiles.ravel()

        results = np.array(
            [
                np.quantile(flat[:, i], flat_a[i], method=q_est_method)
                for i in range(len(flat_a))
            ]
        ).reshape(quantiles.shape)

        return results

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
            0,
            n_instances,
            size=(n_resamples, n_instances),
            dtype=np.intp,
        )

        return resampled_datasets

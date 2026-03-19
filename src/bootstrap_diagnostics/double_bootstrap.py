from dataclasses import dataclass
from typing import Callable, Literal, Any

import numpy as np
import numpy.typing as npt

from joblib import Parallel, delayed

from .resamplers import Resampler


@dataclass(frozen=True)
class ConfidenceInterval:
    """Result of the bootstrap confidence interval procedure."""

    confidence_level: float
    side: Literal["two", "lower", "upper"]

    # Allow both scalar and array confidence intervals
    # (e.g. regression coefficients for multiple linear regression)
    lower: npt.NDArray[np.float64] | float
    upper: npt.NDArray[np.float64] | float

    def __str__(self) -> str:
        return f"lower = {self.lower}\nupper = {self.upper}\nconfidence level = {self.confidence_level}, side = {self.side}"

    # TODO: diagnostic results could potentially be included here?


class DoubleBootstrap:
    """
    Double bootstrap procedure.

    Implements double percentile bootstrap procedure for
    confidence interval construction.

    Note that the CIs for non-scalar statistics are component-wise and do not
    represent a joint confidence region.

    Parameters
    ----------

    statistic : Callable[[npt.ArrayLike], npt.NDArray[Any] | float]
        The function used to calculate the statistic of interest.
        Must follow the signature ``f(data) -> npt.NDArray[Any] | float``.

    resampler : Resampler
        The ``Resampler`` that implements the desired resampling procedure.

    Raises
    ------
    TypeError
        If ``statistic`` is not callable or if ``resampler`` is not a subclass
        of Resampler.
    """

    def __init__(
        self,
        statistic: Callable[[npt.ArrayLike], npt.NDArray[Any] | float],
        resampler: Resampler,
    ) -> None:
        if not isinstance(resampler, Resampler):
            raise TypeError(
                f"resampler must be a subclass of Resampler, got {type(resampler)}"
            )
        if not callable(statistic):
            raise TypeError("Statistic must be callable")

        self._statistic = statistic
        self._resampler = resampler

        # Store the original data sample from the resampler
        self._data_sample = resampler.data_sample

    def confidence_interval(
        self,
        confidence_level: float = 0.95,
        side: Literal["two", "lower", "upper"] = "two",
        B1_resamples: int = 1000,
        B2_resamples: int = 250,
        q_est_method: str = "median_unbiased",
        n_jobs: int = 1,
        seed: int | None = None,
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
            raise ValueError(f"Side must be 'two', 'lower' or 'upper'; got {side}")
        if B1_resamples <= 0 or B2_resamples <= 0:
            raise ValueError("Number of resamples must be non-negative")

        # Use a seed sequence to instantiate high quality
        # probably non-overlapping bit-generators
        ss = np.random.SeedSequence(seed)

        return self._double_percentile_ci(
            confidence_level,
            side,
            B1_resamples,
            B2_resamples,
            q_est_method,
            n_jobs,
            ss,
        )

    def _double_percentile_ci(
        self,
        confidence_level: float,
        side: Literal["two", "lower", "upper"],
        B1: int,
        B2: int,
        q_est_method: str,
        n_jobs: int,
        ss: np.random.SeedSequence,
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
        ss: np.random.SeedSequence
            A seed sequence object which allows for a reproducible way to set the
            initial state for independent and very probably non-overlapping BitGenerators.

        Returns
        -------
        ConfidenceInterval
            The computed bootstrap confidence interval.
        """

        # Evaluate the statistic on the original sample (needed for calibration)
        estimate = self._statistic(self._data_sample)

        # Spawn a sequence of seed sequences used for seeding independent bit-generators
        # We need B1 + 1 of them since we also have to have an rng for the top level
        ss_array = ss.spawn(B1 + 1)

        # Outer bootstrap RNG to resample datasets from the original sample
        # that belongs to self.resampler
        rng_outer = np.random.default_rng(ss_array[0])

        # Derive the seeds for all B1 jobs using the seed sequences computed
        # from the original sequence
        ss_l2 = ss_array[1:]

        # Delegate the tasks
        results = Parallel(n_jobs=n_jobs)(
            delayed(DoubleBootstrap._process_b1)(
                estimate,
                self._resampler.draw_sample(rng_outer),
                self._resampler,
                self._statistic,
                B2,
                ss_l2[i],
            )
            for i in range(B1)
        )

        # zip((1, 2), (3, 4)) = (1, 3), (2, 4)
        # Since we are working with potentially multi-dimensional statistics,
        # we need to store the estimates in an array of the same shape as the
        # statistic output, but with an additional dimension for the
        # number of resamples.
        # For example, if the statistic is a vector of length k, the shape of
        # l1_estimates will be (B1, k) (basically stack them on top of each other)
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
                lower = np.full_like(estimate, -np.inf)
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
                upper = np.full_like(estimate, np.inf)

        return ConfidenceInterval(
            confidence_level,
            side,
            lower,
            upper,
        )

    @staticmethod
    def _process_b1(
        estimate: npt.NDArray[np.float64] | float,
        data_sample: npt.ArrayLike,
        resampler: Resampler,
        statistic: Callable[[npt.ArrayLike], npt.NDArray[Any] | float],
        B2: int,
        ss: np.random.SeedSequence,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Internal method that computes the double bootstrap procedure for a single
        first level resample.

        Parameters
        ----------
        estimate : npt.NDArray[np.float64] | float
            The estimate of the statistic computed from the original sample.
        data_sample: npt.ArrayLike
            The data sample used for sampling the second level bootstrap datasets.
        resampler: Resampler
            The resampler used to construct the second level bootstrap resampler.
        statistic : Callable[[npt.ArrayLike], npt.NDArray[Any] | float]
            The function used to calculate the statistic of interest.
            Must follow the signature ``f(data) -> npt.NDArray[Any] | float``.
        B2 : int
            Number of bootstrap resamples in the second level for calibration
            of the percentile method.
        ss: np.random.SeedSequence
            A seed sequence object which allows for a reproducible way to set the
            initial state for independent and very probably non-overlapping BitGenerators.

        Returns
        -------
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
            The level 1 bootstrap estimate computed from the resample and G^*(hat{theta})
        """

        # Instantiate a local RNG that is unique to this process
        local_rng = np.random.default_rng(ss)

        l1_estimate = statistic(data_sample)

        # Initialize the level 2 resampler with the current level 1 resample as the new "original" dataset
        l2_resampler = resampler.with_data(data_sample)

        # Compute the level 2 estimates
        l2_estimates = np.array(
            [statistic(l2_resampler.draw_sample(local_rng)) for _ in range(B2)]
        )

        # How many level 2 estimates are less than or equal to the original estimate?
        # CDF evaluation G^*(\hat{\theta})
        # Compute the mean along the axis that correspond to the same component
        # of the statistic (e.g. if the statistic is a vector, compute the mean for each component separately)
        cdf_eval = np.mean(l2_estimates <= estimate, axis=0)

        return l1_estimate, cdf_eval

    @staticmethod
    def _quantile_per_component(
        data: npt.NDArray[np.float64],
        quantiles: npt.NDArray[np.float64],
        q_est_method: str,
    ):
        """
        Internal method that computes the per component quantiles of the given data.

        This is nedded for non-scalar statistics, where we want to compute
        the quantiles for each component separately.

        Parameters
        ----------
        data: npt.NDArray[np.float64]
            Data we want to compute per component quantiles for.
        quantiles: npt.NDArray[np.float64]
            Per component quantiles. Need to have the same shape as data[i].
        q_est_method: str
            Method for quantile estimation. Passed as ``numpy.quantile``'s argument ``method``.

        Returns
        -------
        npt.NDArray[np.float64]
            Per component quantiles of the data.
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

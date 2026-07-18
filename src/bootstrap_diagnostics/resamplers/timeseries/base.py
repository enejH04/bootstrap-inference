from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..base import Resampler


class TimeSeriesBlockResampler(Resampler):
    """
    Abstract base class for block bootstrap resampling strategies for time series.

    Parameters
    ----------
    block_length : int
        The length of the blocks, that are resampled with replacement from the
        original time series.

    Raises
    ------
    ValueError
        If ``block_length`` is less than 2 or greater than the number of observations.
    """

    def __init__(
        self,
        data_sample: npt.NDArray | pd.DataFrame,
        block_length: int,
        axis: int = 0,
    ) -> None:
        super().__init__(data_sample, axis)

        # Block length of 1 doesn't make sense since that will lead to IID case
        if block_length < 2:
            raise ValueError(
                f"block_length must be at least 2, got {block_length}"
            )

        n = self.n_observations

        if block_length > n:
            raise ValueError(
                f"block_length cannot exceed the number of observations {n}, got {block_length}"
            )

        self._block_length = block_length

    @property
    def n_observations(self) -> int:
        """int: The total number of observations along the resampling axis."""
        return self._data_sample.shape[self._axis]

    def with_data(
        self,
        new_data_sample: npt.NDArray | pd.DataFrame,
    ) -> Self:
        return type(self)(
            new_data_sample, axis=self._axis, block_length=self._block_length
        )

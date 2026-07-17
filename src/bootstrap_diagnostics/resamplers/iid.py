from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Resampler


class IIDResampler(Resampler):
    """
    Resampler that draws new samples independently and identically distributed (IID)
    from the original dataset.
    """

    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        n_observations = self.data_sample.shape[self._axis]

        # Sample indices with replacement from the original dataset
        return rng.integers(low=0, high=n_observations, size=n_observations)

    def with_data(
        self,
        new_data_sample: npt.NDArray | pd.DataFrame,
    ) -> Self:
        # Use this convention in order to be able to inherit this resampler and
        # only change the draw_sample method
        return type(self)(new_data_sample, axis=self._axis)

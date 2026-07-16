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

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray | pd.DataFrame:
        n_resamples = self.data_sample.shape[self._axis]

        # Sample indices with replacement from the original dataset
        indices = rng.integers(low=0, high=n_resamples, size=n_resamples)

        # Use the sampled indices to create the resampled dataset
        if self._is_dataframe:
            resample = np.take(self._values, indices, axis=self._axis)
            return pd.DataFrame(resample, columns=self._columns, copy=False)

        resample = np.take(self.data_sample, indices, axis=self._axis)

        return resample

    def with_data(
        self,
        new_data_sample: npt.NDArray | pd.DataFrame,
    ) -> Self:
        # Use this convention in order to be able to inherit this resampler and
        # only change the draw_sample method
        return type(self)(new_data_sample, axis=self._axis)

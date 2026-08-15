from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import NonparametricResampler


class IIDResampler(NonparametricResampler):
    """
    Resampler that draws new samples independently and identically distributed (IID)
    from the original dataset.
    """

    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        n_observations = self.data_sample.shape[self._axis]

        # Sample indices with replacement from the original dataset
        return rng.integers(low=0, high=n_observations, size=n_observations)

    def _draw_batch_indices(
        self, b: int, rng: np.random.Generator
    ) -> npt.NDArray:
        """
        Generate indices used for batch resampling. Note that this returns a
        `(b, n_observations)` dimensional array where each row represents one
        resample.

        Parameters
        ----------
        b: int
            Batch size.
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray
            An array of indices with shape (b, n_observations)

        """
        n_observations = self.data_sample.shape[self._axis]

        # Draw a whole batch of indices
        return rng.integers(
            low=0, high=n_observations, size=(b, n_observations)
        )

    @property
    def supports_batching(self) -> bool:
        return isinstance(self._data_sample, np.ndarray)

    def draw_batch_sample(
        self, b: int, rng: np.random.Generator
    ) -> npt.NDArray:
        """
        Generate b resamples in a batch. Note that this returns a
        `(b, data_sample.shape)` array.

        Note that this method only works for NumPy arrays.

        Parameters
        ----------
        b: int
            Batch size.
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray
            A batch resample of shape `(b, data_sample.shape)`.

        Raises
        ------
        ValueError
            If ``b <= 0``.
        TypeError
            If the data sample isn't a NumPy array
        """
        if b <= 0:
            raise ValueError(f"Expected batch size at least 1, got {b}")
        if not isinstance(self._data_sample, np.ndarray):
            raise TypeError("Batch resampling only works with NumPy arrays")

        batch_indices = self._draw_batch_indices(b, rng)
        resample = np.take(
            self._data_sample,
            batch_indices,
            axis=self._axis,
        )
        # In order to preserve batched dimensions, move the dims! (this is due
        # to how np.take handles using matrix indexing). Look at
        # https://numpy.org/doc/stable/reference/generated/numpy.take.html
        return np.moveaxis(resample, self._axis, 0)

    def with_data(
        self,
        new_data_sample: npt.NDArray | pd.DataFrame,
    ) -> Self:
        # Use this convention in order to be able to inherit this resampler and
        # only change the draw_sample method
        return type(self)(new_data_sample, axis=self._axis)

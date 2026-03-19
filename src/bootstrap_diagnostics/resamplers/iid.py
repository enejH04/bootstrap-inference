from typing import Any
import numpy as np
import numpy.typing as npt

from .base import Resampler


class IIDResampler(Resampler):
    """
    Resampler that draws new samples independently and identically distributed (IID)
    from the original dataset.

    Parameters
    ----------
    data_sample : npt.ArrayLike
        A dataset that can be converted to a NumPy array.
    axis : int, optional
        The axis along which new resamples are drawn from the dataset.
        Defaults to 0.
    """

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray[Any]:
        """
        Generate a single bootstrap resample of the data by drawing IID samples
        with replacement from the original dataset.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray[Any]
            A new dataset of the same shape and type as ``self.data_sample``.
        """
        n_resamples = self.data_sample.shape[self.axis]

        # Sample indices with replacement from the original dataset
        indices = rng.integers(low=0, high=n_resamples, size=n_resamples)
        # Use the sampled indices to create the resampled dataset
        resample = np.take(self.data_sample, indices, axis=self.axis)

        return resample

    def with_data(
        self,
        new_data_sample: npt.ArrayLike,
    ) -> "IIDResampler":
        """
        Create a new IIDResampler instance with the same resampling strategy but
        a different input dataset.

        Parameters
        ----------
        new_data_sample : npt.ArrayLike
            A new dataset that can be converted to a NumPy array.

        Returns
        -------
        IIDResampler
            A new IIDResampler instance initialized with the new dataset.
        """
        return IIDResampler(new_data_sample, axis=self.axis)

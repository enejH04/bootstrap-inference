from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd


class IIDResampler:
    """
    Resampler that draws new samples independently and identically distributed (IID)
    from the original dataset.

    Parameters
    ----------
    data_sample : npt.ArrayLike | pd.DataFrame
        A dataset that can be converted to a NumPy array.
    axis : int, optional
        The axis along which new resamples are drawn from the dataset.
        Defaults to 0.


    Raises
    ------
    ValueError
        If any of the following conditions are met:

        - The input ``data_sample`` is empty.
        - The ``axis`` argument is invalid for the given ``data_sample``.
    """

    def __init__(
        self,
        data_sample: npt.NDArray | pd.DataFrame,
        axis: int = 0,
    ) -> None:
        self._data_sample = data_sample
        self._axis = axis

        if self._data_sample.shape[self._axis] == 0:
            raise ValueError(
                "Data sample must have at least one observation along the resampling axis"
            )
        # Allow numpy negative axis indexing, but check that the axis
        # is valid for the given data sample
        if not (-self._data_sample.ndim <= self._axis < self._data_sample.ndim):
            raise ValueError(
                f"Invalid axis {self._axis} for data sample with {self._data_sample.ndim} dimensions"
            )

        # If the data sample is a DataFrame, store the values and columns for faster resampling
        # Cache the data
        # This is needed for the static type checker
        if isinstance(self._data_sample, pd.DataFrame):
            self._is_dataframe = True
            self._values = self._data_sample.values
            self._columns = self._data_sample.columns
        else:
            self._is_dataframe = False

    @property
    def data_sample(self) -> npt.NDArray | pd.DataFrame:
        """
        The original dataset from which resamples are drawn.

        Returns
        -------
        npt.NDArray | pd.DataFrame
            The original dataset as a NumPy array or pandas DataFrame.
        """
        return self._data_sample

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray | pd.DataFrame:
        """
        Generate a single bootstrap resample of the data by drawing IID samples
        with replacement from the original dataset.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray | pd.DataFrame
            A new dataset of the same shape and type as ``self.data_sample``.
        """
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
    ) -> "IIDResampler":
        """
        Create a new ``IIDResampler`` instance with the same resampling strategy but
        a different input dataset.

        Parameters
        ----------
        new_data_sample : npt.NDArray | pd.DataFrame
            A new dataset.

        Returns
        -------
        IIDResampler
            A new ``IIDResampler`` instance initialized with the new dataset.
        """
        return IIDResampler(new_data_sample, axis=self._axis)

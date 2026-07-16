from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd


class Resampler(ABC):
    """
    Abstract base class for bootstrap resampling strategies.

    Parameters
    ----------
    data_sample : npt.ArrayLike | pd.DataFrame
        The input dataset used for resampling.
    axis : int, optional
        The axis along which new resamples are drawn from the dataset.
        Defaults to 0.

    Raises
    ------
    ValueError
        If any of the following conditions are met:
        - The input ``data_sample`` is empty.
        - The ``axis`` argument is invalid for the given ``data_sample``.

    Notes
    -----
    Implementations must implement the `draw_sample` and `with_data` methods.
    Resampled datasets must be compatible with the statistic being evaluated
    on the dataset.
    """

    def __init__(
        self,
        data_sample: npt.NDArray | pd.DataFrame,
        axis: int = 0,
    ) -> None:
        self._data_sample = data_sample
        self._axis = axis

        # Allow numpy negative axis indexing, but check that the axis
        # is valid for the given data sample
        if not (-self._data_sample.ndim <= self._axis < self._data_sample.ndim):
            raise ValueError(
                f"Invalid axis {self._axis} for data sample with {self._data_sample.ndim} dimensions"
            )
        if self._data_sample.shape[self._axis] == 0:
            raise ValueError(
                "Data sample must have at least one observation along the resampling axis"
            )

    @property
    def data_sample(self) -> npt.NDArray | pd.DataFrame:
        """
        The original dataset from which resamples are drawn.

        Returns
        -------
        npt.NDArray | pd.DataFrame
            The original dataset as a NumPy array.
        """
        return self._data_sample

    @abstractmethod
    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray | pd.DataFrame:
        """
        Generate a single bootstrap resample of the data.

        To implement a custom resampling procedure,
        (hierarchical, block, ...) subclasses must override this method.

        To ensure that the results are fully reproducible, use the provided
        NumPy random number generator for all random operations.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray | pd.DataFrame
            A new dataset of the same shape and type as ``self.data_sample``.
        """
        ...

    # Note that this is needed to allow for different __init__ arguments
    @abstractmethod
    def with_data(self, new_data_sample: Any) -> Self:
        """
        Create a new resampler instance with the same resampling strategy but
        a different input dataset.

        The main use of this method is to allow the double bootstrap procedure
        to reuse the same resampling strategy for both the outer and inner bootstrap
        loops and thus allow as much flexibility as possible.

        Parameters
        ----------
        new_data_sample : Any
            A new dataset.

        Returns
        -------
        Resampler
            A new resampler instance with the same resampling strategy but
            a different input dataset.
        """
        ...

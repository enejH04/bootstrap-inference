from abc import ABC, abstractmethod
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.array_utils import normalize_axis_index


# So we can use isinstance check
@runtime_checkable
class BatchResampler(Protocol):
    def draw_batch_sample(
        self, b: int, rng: np.random.Generator
    ) -> npt.NDArray:
        """
        Generate a batch bootstrap resample of the data with shape
        ``(b, *data_sample.shape)``.

        To ensure that the results are fully reproducible, use the provided
        NumPy random number generator for all random operations.

        Parameters
        ----------
        b : int
            Batch size.
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray
            A batch of resampled datasets of size ``(b, *data_sample.shape)``.
        """
        ...


class Resampler(ABC):
    """
    Abstract base class for bootstrap resampling strategies.

    Parameters
    ----------
    data_sample : npt.ArrayLike | pd.DataFrame
        The input dataset used for resampling.
    axis : int, optional
        The axis along which new resamples are drawn from the dataset. For Pandas
        DataFrames, only row resampling (``axis = 0``) is supported. Defaults to 0.

    Raises
    ------
    ValueError
        If any of the following conditions are met:
        - The input ``data_sample`` is empty.
        - The ``axis`` argument is invalid for the given ``data_sample``.
        - If ``axis != 0`` and ``data_sample`` is a Pandas DataFrame.

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
        # Normalize axis using internal NumPy functions
        self._axis = normalize_axis_index(axis, self._data_sample.ndim)

        if self._data_sample.shape[self._axis] == 0:
            raise ValueError(
                "Data sample must have at least one observation along the resampling axis"
            )
        if isinstance(self._data_sample, pd.DataFrame) and self._axis != 0:
            raise ValueError(
                f"For Pandas DataFrames, only row resampling (axis = 0) is supported; got {self._axis}"
            )

    @property
    def axis(self) -> int:
        """
        int : The axis along which resamples are drawn
        """
        return self._axis

    @property
    def data_sample(self) -> npt.NDArray | pd.DataFrame:
        """
        npt.NDArray | pd.DataFrame : The original dataset from which resamples are drawn.
        """
        return self._data_sample

    @property
    def supports_batching(self) -> bool:
        """
        bool : Whether the resampler supports batch resampling
        """
        return False

    @abstractmethod
    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray | pd.DataFrame:
        """
        Generate a single bootstrap resample of the data.

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


class NonparametricResampler(Resampler):
    """
    Base class for nonparametric bootstrap resampling strategies.
    """

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray | pd.DataFrame:
        indices = self._draw_indices(rng)

        # Use the sampled indices to create the resampled dataset
        # For dataframe we resample the rows
        if isinstance(self._data_sample, pd.DataFrame):
            return self._data_sample.iloc[indices].reset_index(drop=True)

        resample = np.take(self._data_sample, indices, axis=self._axis)

        return resample

    @abstractmethod
    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        """
        Generate indices for a single bootstrap resample of the data.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray
            An array of indices used to resample from the dataset.
        """
        ...

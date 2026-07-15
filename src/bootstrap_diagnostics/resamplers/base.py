from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class Resampler(Protocol):
    """
    Protocol for bootstrap resampling strategies.

    Notes
    -----
    Implementations must implement the `draw_sample` and `with_data` methods.
    Resampled datasets must be compatible with the statistic being evaluated
    on the dataset.
    """

    @property
    def data_sample(self) -> npt.NDArray | pd.DataFrame:
        """
        The original dataset from which resamples are drawn.

        Returns
        -------
        npt.NDArray | pd.DataFrame
            The original dataset as a NumPy array.
        """
        ...

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
    def with_data(
        self, new_data_sample: npt.ArrayLike | pd.DataFrame
    ) -> "Resampler":
        """
        Create a new resampler instance with the same resampling strategy but
        a different input dataset.

        The main use of this method is to allow the double bootstrap procedure
        to reuse the same resampling strategy for both the outer and inner bootstrap
        loops and thus allow as much flexibility as possible.

        Parameters
        ----------
        new_data_sample : npt.ArrayLike | pd.DataFrame
            A new dataset.

        Returns
        -------
        Resampler
            A new resampler instance with the same resampling strategy but
            a different input dataset.
        """
        ...

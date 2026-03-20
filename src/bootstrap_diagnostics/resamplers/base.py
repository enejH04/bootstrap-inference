from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Resampler(ABC):
    """
    Abstract base class for bootstrap resampling strategies.

    Parameters
    ----------
    data_sample : npt.ArrayLike
        A dataset that can be converted to a NumPy array.
    axis : int, optional
        The axis along which resampling from `data_sample` is performed (e.g. 0 for rows, 1 for columns).
        Defaults to 0.

    Raises
    ------
    ValueError
        If any of the following conditions are met:

        - The input data sample cannot be converted to a NumPy array.
        - The input data sample is empty.
        - The axis argument is invalid for the given data sample.

    Notes
    -----
    Subclasses must implement the `draw_sample` and `with_data` methods.
    Resampled datasets must be compatible with the statistic being evaluated
    on the dataset.
    """

    def __init__(
        self,
        data_sample: npt.ArrayLike,
        axis: int = 0,
    ) -> None:
        try:
            # TODO: add a wrapper for pandas data frames - this actually already
            # works as long as the data frame can be converted to a Numpy array of floats
            self.data_sample = np.asarray(data_sample)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "Cannot convert given array of data points to a Numpy array"
            ) from e
        if self.data_sample.size == 0:
            raise ValueError(
                "Input data sample is empty. Cannot perform bootstrap"
            )
        # Allow numpy negative axis indexing, but check that the axis
        # is valid for the given data sample
        if not (-self.data_sample.ndim <= axis < self.data_sample.ndim):
            raise ValueError(
                f"Invalid axis {axis} for data sample with {self.data_sample.ndim} dimensions"
            )
        self.axis = axis

    @abstractmethod
    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray:
        """
        Generate a single bootstrap resample of the data.

        This is an abstract method. To implement a custom resampling procedure,
        (hierarchical, block, ...) subclasses must override this method.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random number generator.

        Returns
        -------
        npt.NDArray
            A new dataset of the same shape and type as ``self.data_sample``.
        """
        ...

    # Note that this is needed to allow for different __init__ arguments
    @abstractmethod
    def with_data(self, new_data_sample: npt.ArrayLike) -> "Resampler":
        """
        Create a new resampler instance with the same resampling strategy but
        a different input dataset.

        The main use of this method is to allow the double bootstrap procedure
        to reuse the same resampling strategy for both the outer and inner bootstrap
        loops and thus allow as much flexibility as possible.

        Parameters
        ----------
        new_data_sample : npt.ArrayLike
            A new dataset that can be converted to a NumPy array.

        Returns
        -------
        Resampler
            A new resampler instance with the same resampling strategy but
            a different input dataset.
        """
        ...

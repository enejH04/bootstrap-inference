from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Resampler(ABC):
    """
    Abstract base class for bootstrap resampling strategies.

    Parameters
    ----------
    data_sample : npt.ArrayLike
        A dataset that can be converted to a NumPy array of floats.
    axis : int, optional
        The axis along which new resamples are drawn from the dataset.
        Defaults to 0.
    """

    def __init__(
        self,
        data_sample: npt.ArrayLike,
        axis: int = 0,
    ) -> None:
        try:
            # Try to convert the input data sample to a Numpy array of floats
            # TODO: add a wrapper for pandas data frames
            self.data_sample = np.asarray(data_sample, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "Cannot convert given array of data points to a Numpy float array"
            ) from e
        if self.data_sample.size == 0:
            raise ValueError("Input data sample is empty. Cannot perform bootstrap")
        self.axis = axis
        self.n_obs = self.data_sample.shape[axis]

    @abstractmethod
    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """
        Generate a single bootstrap resample of the data.

        This is an abstract method. To implement a custom resampling procedure,
        (hierarchical, block, ...) subclasses must override this method.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray[np.float64]
            A new dataset of the same shape as ``self.data_sample``.
        """
        ...

    # Note that this is needed to allow for different __init__ arguments
    # in e.g. block resampling strategies
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
            A new dataset that can be converted to a NumPy array of floats.

        Returns
        -------
        Resampler
            A new resampler instance with the same resampling strategy but
            a different input dataset.
        """
        ...

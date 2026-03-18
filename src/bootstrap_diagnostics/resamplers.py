from abc import ABC, abstractmethod

import numpy.typing as npt

import numpy as np


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
        except (ValueError, TypeError):
            raise ValueError(
                "Cannot convert given array of data points to a Numpy float array"
            )
        if self._data_sample.size == 0:
            raise ValueError(
                "Input data sample is empty. Cannot perform bootstrap"
            )
        self.axis = axis
        self.n_obs = self.data_sample.shape[axis]

    @abstractmethod
    def draw_sample(self, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """
        Generate a single bootstrap resample of the data.

        This is an abstract method. To implement a custom resampling procedure,
        (hierarchical, block, ...) subclasses must override this method.


        Parameters
        ----------
        rng : Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray[np.float64]
            A new dataset of the same shape as ``self.data_sample``.
        """
        raise NotImplementedError

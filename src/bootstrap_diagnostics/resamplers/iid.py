import numpy as np
import numpy.typing as npt


class IIDResampler:
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

    def __init__(
        self,
        data_sample: npt.ArrayLike,
        axis: int = 0,
    ) -> None:
        try:
            # TODO: add a wrapper for pandas data frames - this actually already
            # works as long as the data frame can be converted to a Numpy array of floats
            self._data_sample = np.asarray(data_sample)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "Cannot convert given array of data points to a NumPy array"
            ) from e
        if self._data_sample.size == 0:
            raise ValueError(
                "Input data sample is empty. Cannot perform bootstrap"
            )
        # Allow numpy negative axis indexing, but check that the axis
        # is valid for the given data sample
        if not (-self._data_sample.ndim <= axis < self._data_sample.ndim):
            raise ValueError(
                f"Invalid axis {axis} for data sample with {self._data_sample.ndim} dimensions"
            )
        self.axis = axis

    @property
    def data_sample(self) -> npt.NDArray:
        """
        The original dataset from which resamples are drawn.

        Returns
        -------
        npt.NDArray
            The original dataset as a NumPy array.
        """
        return self._data_sample

    def draw_sample(
        self,
        rng: np.random.Generator,
    ) -> npt.NDArray:
        """
        Generate a single bootstrap resample of the data by drawing IID samples
        with replacement from the original dataset.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random number generator.

        Returns
        -------
        npt.NDArray
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
        Create a new ``IIDResampler`` instance with the same resampling strategy but
        a different input dataset.

        Parameters
        ----------
        new_data_sample : npt.ArrayLike
            A new dataset that can be converted to a NumPy array.

        Returns
        -------
        IIDResampler
            A new ``IIDResampler`` instance initialized with the new dataset.
        """
        return IIDResampler(new_data_sample, axis=self.axis)

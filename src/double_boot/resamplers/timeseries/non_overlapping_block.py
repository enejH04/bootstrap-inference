import numpy as np
import numpy.typing as npt

from .base import TimeSeriesBlockResampler


class NonOverlappingBlockResampler(TimeSeriesBlockResampler):
    """
    Generate bootstrap sample indices according to the non-overlapping block resampling strategy.

    Given n observations and block length l, we have floor(n / l) possible starting points.
    We sample ceil(n / l) blocks with replacement by randomly choosing their starting indices
    from the range [0 ... floor(n / l) - 1] * l. Suppose we have sampled index i, then the sampled block
    is of the form [i, i + 1, ..., i + l - 1].


    These blocks are then pasted together and truncated at length n to match the original time series size.
    """

    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        n = self.n_observations

        n_sampled_blocks = int(np.ceil(n / self._block_length))
        n_available_blocks = int(np.floor(n / self._block_length))

        # Compute starting points of the blocks
        starts = (
            rng.integers(
                n_available_blocks,
                size=n_sampled_blocks,
            )
            * self._block_length
        )

        block_offsets = np.arange(self._block_length)

        # broadcast them together -> n_sampled_blocks x 1 + 1 x self._block_length
        indices = starts[:, np.newaxis] + block_offsets

        # Flatten and truncate
        return indices.ravel()[:n]

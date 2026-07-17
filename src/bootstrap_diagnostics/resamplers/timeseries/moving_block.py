import numpy as np
import numpy.typing as npt

from .base import TimeSeriesBlockResampler


class MovingBlockResampler(TimeSeriesBlockResampler):
    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        n = self.n_observations

        n_available_blocks = n - self._block_length + 1
        n_sampled_blocks = int(np.ceil(n / self._block_length))

        # Compute starting points of the blocks
        starts = rng.integers(
            n_available_blocks,
            size=n_sampled_blocks,
        )

        block_offsets = np.arange(self._block_length)

        # broadcast them together -> n_sampled_blocks x 1 + 1 x self._block_length
        indices = starts[:, np.newaxis] + block_offsets

        # Flatten and truncate
        return indices.ravel()[:n]

import numpy as np
import numpy.typing as npt

from .base import TimeSeriesBlockResampler


class StationaryBlockResampler(TimeSeriesBlockResampler):
    """
    Generate bootstrap sample indices according to the stationary block resampling strategy.

    Given n observations and expected block length l, we have n possible starting points.
    We sample blocks until the resampled time series is as long or longer than the
    original. Blocks lengths aren't fixed but rather sampled from a Geometric(1 / l) distribution.

    At the end, the blocks are pasted together and truncated at length n to match the original time series size.
    """

    def _draw_indices(self, rng: np.random.Generator) -> npt.NDArray:
        n = self.n_observations
        p = 1 / self._block_length

        out = np.empty(n, dtype=np.intp)
        idx = 0

        while idx < n:
            start = rng.integers(n)
            block_length = rng.geometric(p)

            # Allowed size
            allowed_size = min(n - idx, block_length)
            out[idx : idx + allowed_size] = (
                np.arange(start, start + allowed_size) % n
            )
            idx += allowed_size

        return out

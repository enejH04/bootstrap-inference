import numpy as np
import pandas as pd
import pytest


def test_resample_preserves_shape(resampler, rng):
    sample = resampler.draw_sample(rng)

    assert sample.shape == resampler.data_sample.shape


def test_draw_sample_type(resampler, rng):
    sample = resampler.draw_sample(rng)

    assert type(sample) == type(resampler.data_sample)


def test_draw_sample_reproducible(resampler):
    random_seed = 0

    sample1 = resampler.draw_sample(np.random.default_rng(random_seed))
    sample2 = resampler.draw_sample(np.random.default_rng(random_seed))

    if isinstance(sample1, pd.DataFrame):
        pd.testing.assert_frame_equal(sample1, sample2)
    else:
        np.testing.assert_array_equal(sample1, sample2)

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


def test_hierarchical_resample_reproducible(hierarchical_resampler):
    random_seed = 0

    sample1 = hierarchical_resampler.draw_sample(
        np.random.default_rng(random_seed)
    )
    sample2 = hierarchical_resampler.draw_sample(
        np.random.default_rng(random_seed)
    )

    pd.testing.assert_frame_equal(sample1, sample2)


def test_hierarchical_logic(hierarchical_resampler, hierarchical_data, rng):
    sample = hierarchical_resampler.draw_sample(rng)

    assert isinstance(sample, pd.DataFrame)
    assert list(sample.columns) == list(hierarchical_data.columns)

    original_sizes = hierarchical_data.groupby("school", sort=False).size()
    sampled_sizes = sample.groupby("school", sort=False).size()

    for school in sampled_sizes.index:
        # The number of students in the sampled school should be a multiple of the original
        # (if School A was drawn twice, it should have 3 * 2 = 6 rows).
        assert sampled_sizes[school] % original_sizes[school] == 0

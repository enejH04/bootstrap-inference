import numpy as np
import pandas as pd


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


def test_block_resampler_strategy(block_resampler, rng):
    block_length = block_resampler._block_length
    indices = block_resampler._draw_indices(rng)
    n = block_resampler.n_observations

    # First check the untruncated part
    complete_blocks_len = (n // block_length) * block_length
    clean_indices = indices[:complete_blocks_len]

    blocks = clean_indices.reshape(-1, block_length)

    intra_bloock_diffs = np.diff(blocks, axis=1) % n

    assert np.all(intra_bloock_diffs == 1), (
        "Elements within blocks are not sequential"
    )

    # Check remainder
    remainder = indices[complete_blocks_len:]
    remaining_diffs = np.diff(remainder) % n

    if complete_blocks_len != n:
        assert np.all(remaining_diffs == 1), (
            "Elements in the truncated block aren't sequential"
        )


def test_non_overlapping_block_strategy(non_overlapping_block_resampler, rng):
    block_length = non_overlapping_block_resampler._block_length
    indices = non_overlapping_block_resampler._draw_indices(rng)
    n = non_overlapping_block_resampler.n_observations

    complete_blocks_len = (n // block_length) * block_length

    clean_indices = indices[:complete_blocks_len].reshape(-1, block_length)

    # check that the starting points make sense -> multiple of block lengths
    remainders = clean_indices[:, 0] % block_length
    assert np.all(remainders == 0), (
        "Starting points of the blocks aren't multiples of block length"
    )

    if complete_blocks_len < n:
        assert indices[complete_blocks_len] % block_length == 0, (
            "Starting points of the blocks aren't multiples of block length"
        )

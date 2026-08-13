import numpy as np
import pytest

from double_boot import Bootstrap, IIDResampler

SEED = 44

# Different possible configurations of the bootstrap estimators
# (vectorized, n_jobs)
EXECUTION_MODES = [
    pytest.param(False, 1, id="unbatched-serial"),
    pytest.param(False, 5, id="unbatched-parallel"),
    pytest.param(True, 1, id="batched-serial"),
    pytest.param(True, 5, id="batched-parallel"),
]


def _bootstrap(sample, vectorized):
    return Bootstrap(
        np.mean,
        IIDResampler(sample),
        vectorized=vectorized,
    )


def _assert_ci_equal(
    result,
    expected,
):
    np.testing.assert_allclose(result.estimate, expected.estimate)
    np.testing.assert_allclose(result.lower, expected.lower)
    np.testing.assert_allclose(result.upper, expected.upper)

    assert result.confidence_level == expected.confidence_level
    assert result.side == expected.side


@pytest.mark.parametrize("vectorized,n_jobs", EXECUTION_MODES)
def test_bias_consistent_across_execution_modes(
    sample,
    vectorized,
    n_jobs,
):
    # Without vectorization and parallelism
    expected = _bootstrap(sample, vectorized=False).bias(
        n_resamples=50,
        n_jobs=1,
        seed=SEED,
    )

    result = _bootstrap(sample, vectorized=vectorized).bias(
        n_resamples=50,
        n_jobs=n_jobs,
        seed=SEED,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("vectorized,n_jobs", EXECUTION_MODES)
def test_variance_consistent_across_execution_modes(
    sample,
    vectorized,
    n_jobs,
):
    # Without vectorization and parallelism
    expected = _bootstrap(sample, vectorized=False).variance(
        n_resamples=50,
        n_jobs=1,
        seed=SEED,
    )

    result = _bootstrap(sample, vectorized=vectorized).variance(
        n_resamples=50,
        n_jobs=n_jobs,
        seed=SEED,
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("side", ["two", "lower", "upper"])
@pytest.mark.parametrize("vectorized,n_jobs", EXECUTION_MODES)
def test_percentile_ci_consistent_across_execution_modes(
    sample,
    vectorized,
    n_jobs,
    side,
):
    # No vectorization and parallelism
    expected = _bootstrap(sample, vectorized=False).percentile_ci(
        confidence_level=0.95,
        side=side,
        b_resamples=50,
        n_jobs=1,
        seed=SEED,
    )

    # Different levels of vectorization and parallelism
    result = _bootstrap(sample, vectorized=vectorized).percentile_ci(
        confidence_level=0.95,
        side=side,
        b_resamples=50,
        n_jobs=n_jobs,
        seed=SEED,
    )

    _assert_ci_equal(result, expected)


@pytest.mark.parametrize("side", ["two", "lower", "upper"])
@pytest.mark.parametrize("vectorized,n_jobs", EXECUTION_MODES)
def test_double_percentile_ci_consistent_across_execution_modes(
    sample,
    vectorized,
    n_jobs,
    side,
):
    # No vectorization and parallelism
    expected = _bootstrap(sample, vectorized=False).double_percentile_ci(
        confidence_level=0.95,
        side=side,
        b1_resamples=20,
        b2_resamples=15,
        n_jobs=1,
        seed=SEED,
    )

    # Different levels of vectorization and parallelism
    result = _bootstrap(sample, vectorized=vectorized).double_percentile_ci(
        confidence_level=0.95,
        side=side,
        b1_resamples=20,
        b2_resamples=15,
        n_jobs=n_jobs,
        seed=SEED,
    )

    _assert_ci_equal(result, expected)


def test_double_percentile_ci_uses_cached_resamples(sample):
    bootstrap = _bootstrap(sample, vectorized=False)

    # Fill the cache with the given seed
    bootstrap.double_percentile_ci(
        confidence_level=0.9,
        b1_resamples=20,
        b2_resamples=15,
        seed=SEED,
    )

    # Use the cached values to compute the double percentile CI
    result = bootstrap.double_percentile_ci(
        confidence_level=0.95,
        b1_resamples=20,
        b2_resamples=15,
        use_cached=True,
    )

    # Recompute the double percentile CI (no caching)
    expected = _bootstrap(sample, vectorized=False).double_percentile_ci(
        confidence_level=0.95,
        b1_resamples=20,
        b2_resamples=15,
        seed=SEED,
    )

    # We expect the intervals to match
    _assert_ci_equal(result, expected)


def test_double_percentile_ci_rejects_missing_cache(sample):
    bootstrap = _bootstrap(sample, vectorized=False)

    with pytest.raises(ValueError, match="No cached bootstrap estimates"):
        bootstrap.double_percentile_ci(
            b1_resamples=20,
            b2_resamples=15,
            use_cached=True,
        )


def test_percentile_ci_uses_cached_resamples(sample):
    bootstrap = _bootstrap(sample, vectorized=False)

    # Fill the cache with the given seed
    bootstrap.percentile_ci(
        confidence_level=0.9,
        b_resamples=20,
        seed=SEED,
    )

    # Use the cached values to compute the percentile CI
    result = bootstrap.percentile_ci(
        confidence_level=0.95,
        b_resamples=20,
        use_cached=True,
    )

    # Recompute the percentile CI (no caching)
    expected = _bootstrap(sample, vectorized=False).percentile_ci(
        confidence_level=0.95,
        b_resamples=20,
        seed=SEED,
    )

    # We expect the intervals to match
    _assert_ci_equal(result, expected)


def test_percentile_ci_rejects_missing_cache(sample):
    bootstrap = _bootstrap(sample, vectorized=False)

    with pytest.raises(ValueError, match="No cached bootstrap estimates"):
        bootstrap.percentile_ci(
            confidence_level=0.95,
            b_resamples=20,
            use_cached=True,
        )


def _column_mean(data):
    return np.mean(data, axis=0)


def _axis_mean(data, axis):
    return np.mean(data, axis=axis)


def test_vector_valued_statistic_preserves_shape():
    data = np.arange(120.0).reshape(30, 4)

    expected = Bootstrap(
        _column_mean,
        IIDResampler(data),
        vectorized=False,
    ).double_percentile_ci(
        b1_resamples=20,
        b2_resamples=15,
        seed=SEED,
    )

    result = Bootstrap(
        _axis_mean,
        IIDResampler(data),
        vectorized=True,
    ).double_percentile_ci(
        b1_resamples=20,
        b2_resamples=15,
        seed=SEED,
    )

    assert result.estimate.shape == (4,)  # ty:ignore[unresolved-attribute]
    assert result.lower.shape == (4,)  # ty:ignore[unresolved-attribute]
    assert result.upper.shape == (4,)  # ty:ignore[unresolved-attribute]
    _assert_ci_equal(result, expected)

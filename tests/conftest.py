from itertools import product

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from bootstrap_diagnostics import (
    CircularBlockResampler,
    HierarchicalResampler,
    IIDResampler,
    MovingBlockResampler,
    NonOverlappingBlockResampler,
    StationaryBlockResampler,
)


# Test with different array shapes
@pytest.fixture(
    params=[
        np.arange(100),
        np.ones((100, 5)),
        np.ones((100, 100, 5)),
        pd.DataFrame({"x": np.arange(100), "y": np.arange(100)}),
    ],
    ids=["1d-array", "2d-array", "3d-array", "data-frame"],
)
def data(request):
    return request.param


@pytest.fixture
def hierarchical_data():
    return pd.DataFrame(
        {
            "school": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "classroom": [1, 1, 2, 1, 2, 1, 1, 2],
            "student_score": [85, 90, 88, 70, 72, 95, 91, 99],
        }
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(
    params=[
        pytest.param(
            lambda data: IIDResampler(data),
            id="IIDResampler",
        ),
        pytest.param(
            lambda data: MovingBlockResampler(data, 3),
            id="MovingBlockResampler",
        ),
        pytest.param(
            lambda data: CircularBlockResampler(data, 3),
            id="CircularBlockResampler",
        ),
        pytest.param(
            lambda data: NonOverlappingBlockResampler(data, 3),
            id="NonOverlappingBlockResampler",
        ),
        pytest.param(
            lambda data: StationaryBlockResampler(data, 3),
            id="StationaryBlockResampler",
        ),
    ],
)
def resampler(request, data):
    return request.param(data)


@pytest.fixture
def hierarchical_resampler(hierarchical_data):
    # Sample schools with replacement, but keep all classrooms and students exactly as they are
    return HierarchicalResampler(
        hierarchical_data,
        hierarchy=[("school", True), ("classroom", False)],
        observation_replacement=False,
    )


BLOCK_RESAMPLERS = [
    MovingBlockResampler,
    CircularBlockResampler,
    NonOverlappingBlockResampler,
]
BLOCK_LENGTHS = [2, 5, 8, 13, 20]


@pytest.fixture(
    params=list(product(BLOCK_RESAMPLERS, BLOCK_LENGTHS)),
    ids=lambda p: f"{p[0].__name__}, l={p[1]}",
)
def block_resampler(request, data):
    return request.param[0](data, block_length=request.param[1])


@pytest.fixture(
    params=BLOCK_LENGTHS, ids=lambda p: f"NonOverlappingBlockResampler, l={p}"
)
def non_overlapping_block_resampler(request):
    return NonOverlappingBlockResampler(
        np.arange(100),
        block_length=request.param,
    )


@pytest.fixture
def sample():
    return np.random.default_rng(123).normal(size=40)

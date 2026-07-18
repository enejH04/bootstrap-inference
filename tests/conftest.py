import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from bootstrap_diagnostics import (
    HierarchicalResampler,
    IIDResampler,
    MovingBlockResampler,
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
            id="iid",
        ),
        pytest.param(
            lambda data: MovingBlockResampler(data, 3),
            id="mb",
        ),
    ]
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


@pytest.fixture(params=[2, 5, 8, 13, 20])
def moving_block_resampler(request, data):
    return MovingBlockResampler(
        data,
        block_length=request.param,
    )

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from bootstrap_diagnostics import IIDResampler


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
def rng():
    return np.random.default_rng(0)


@pytest.fixture(
    params=[
        pytest.param(
            lambda data: IIDResampler(data),
            id="iid",
        )
    ]
)
def resampler(request, data):
    return request.param(data)

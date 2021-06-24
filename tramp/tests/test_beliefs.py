import pytest
import numpy as np
from numpy.testing import assert_allclose
from tramp.beliefs import (
    binary, exponential, mixture,
    normal, positive, sparse, truncated
)
from tramp.checks.check_gradients import check_belief_grad_b, EPSILON


@pytest.mark.parametrize("belief,kwargs", [
    (binary, {}),
    (exponential, {}),
    (normal, {"a": 1}),
    (sparse, {"a": 1, "eta": 1}),
    (positive, {"a": 1}),
    (truncated, {"a": 1, "xmin": -1, "xmax": +1}),
    (mixture, {"a": np.ones(2), "b0": np.array([-1, +1]), "eta":np.ones(2)}),
])
def test_belief_grad_b(belief, kwargs):
    df = check_belief_grad_b(belief, **kwargs)
    assert_allclose(df["r"], df["A1"], rtol=0, atol=EPSILON)
    assert_allclose(df["v"], df["A2"], rtol=0, atol=EPSILON)

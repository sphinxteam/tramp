import pytest
import numpy as np
from numpy.testing import assert_allclose
from tramp.priors import (
    GaussianPrior, GaussBernoulliPrior, BinaryPrior,
    MAP_LaplacePrior, MAP_L21NormPrior,
    ExponentialPrior, PositivePrior, GaussianMixturePrior
)
from tramp.checks.check_limits import check_prior_BO_limit, check_prior_BN_limit
from tramp.checks.check_gradients import (
    check_prior_grad_BO, check_prior_grad_FG, check_prior_grad_RS,
    check_prior_grad_BO_BN, check_prior_grad_EP_scalar, EPSILON
)


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=(10, 10, 10), isotropic=False),
    BinaryPrior(size=100, isotropic=False)
])
def test_separable_prior_vectorization(prior):
    assert not prior.isotropic
    N = np.prod(prior.size)
    ax = np.linspace(1, 2, N)
    ax = ax.reshape(prior.size)
    bx = np.linspace(-2, 2, N)
    bx = bx.reshape(prior.size)
    rx, vx = prior.compute_forward_posterior(ax, bx)
    assert rx.shape == bx.shape
    assert vx.shape == bx.shape
    # check rx vectorization
    rx_ = np.array([prior.scalar_forward_mean(a, b) for a, b in zip(ax, bx)])
    rx_ = rx_.reshape(prior.size)
    assert_allclose(rx, rx_)
    # check vx vectorization
    vx_ = np.array([prior.scalar_forward_variance(a, b) for a, b in zip(ax, bx)])
    vx_ = vx_.reshape(prior.size)
    assert_allclose(vx, vx_)
    # check A vectorization
    A = prior.compute_log_partition(ax, bx)
    A_ = np.mean([prior.scalar_log_partition(a, b) for a, b in zip(ax, bx)])
    assert A == A_


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    BinaryPrior(size=None)
])
def test_prior_BO_limit(prior):
    df = check_prior_BO_limit(prior)
    assert_allclose(df["A_BO"], df["A_RS"])
    assert_allclose(df["vx_BO"], df["vx_RS"])
    assert_allclose(df["mx_BO"], df["mx_RS"])
    assert_allclose(df["mx_BO"], df["qx_RS"])


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    GaussBernoulliPrior(size=None)
])
def test_prior_BN_limit(prior):
    df = check_prior_BN_limit(prior)
    assert_allclose(df["A_FG"], df["A_BN"])
    assert_allclose(df["vx_FG"], df["vx_BN"])
    assert_allclose(df["tx_FG"], df["tx_BN"])


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    GaussBernoulliPrior(size=None)
])
def test_prior_grad_BO(prior):
    df = check_prior_grad_BO(prior)
    assert_allclose(df["mx"], 2*df["grad_mx_hat_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    GaussBernoulliPrior(size=None)
])
def test_prior_grad_FG(prior):
    df = check_prior_grad_FG(prior)
    assert_allclose(df["tx"], -2*df["grad_tx_hat_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    GaussBernoulliPrior(size=None)
])
def test_prior_grad_BO_BN(prior):
    df = check_prior_grad_BO_BN(prior)
    assert_allclose(df["mx"], 2*df["grad_ax_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["vx"], 2*df["grad_ax_I"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("prior", [
    GaussianPrior(size=None),
    GaussBernoulliPrior(size=None)
])
def test_prior_grad_EP_scalar(prior):
    df = check_prior_grad_EP_scalar(prior)
    assert_allclose(df["rx"], df["grad_bx_A1"], rtol=0, atol=EPSILON)
    assert_allclose(df["vx"], df["grad_bx_A2"], rtol=0, atol=EPSILON)
    assert_allclose(df["tx"], -2*df["grad_ax_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("teacher,student", [
    (GaussianPrior(size=None), BinaryPrior(size=None)),
    (GaussBernoulliPrior(size=None), BinaryPrior(size=None)),
])
def test_prior_grad_RS(teacher, student):
    df = check_prior_grad_RS(teacher, student)
    assert_allclose(df["mx"], df["grad_mx_hat_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["qx"], -2*df["grad_qx_hat_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["tx"], -2*df["grad_tx_hat_A"], rtol=0, atol=EPSILON)

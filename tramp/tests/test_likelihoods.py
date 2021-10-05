import pytest
import numpy as np
from numpy.testing import assert_allclose
from tramp.likelihoods import (
    GaussianLikelihood, AbsLikelihood, SgnLikelihood,
    ReluLikelihood, LeakyReluLikelihood, HardTanhLikelihood,
    HardSigmoidLikelihood, SymmetricDoorLikelihood,
    ModulusLikelihood
)
from tramp.checks.check_limits import check_likelihood_BO_limit, check_likelihood_BN_limit
from tramp.checks.check_gradients import (
    check_likelihood_grad_BO, check_likelihood_grad_BO_BN,
    check_likelihood_grad_RS, check_likelihood_grad_FG,
    check_likelihood_grad_EP_scalar, check_likelihood_grad_EP_diagonal,
    EPSILON
)
from tramp.utils.misc import relu, leaky_relu, hard_tanh, hard_sigm, symm_door


def create_likelihoods():
    z = np.linspace(-3, 3, 100)
    return [
        GaussianLikelihood(y=z, isotropic=False),
        AbsLikelihood(y=np.abs(z), isotropic=False),
        SgnLikelihood(y=np.sign(z), isotropic=False),
        ReluLikelihood(y=relu(z), isotropic=False),
        LeakyReluLikelihood(y=leaky_relu(z, 0.1), slope=0.1, isotropic=False),
        HardTanhLikelihood(y=hard_tanh(z), isotropic=False),
        HardSigmoidLikelihood(y=hard_sigm(z), isotropic=False),
        SymmetricDoorLikelihood(y=symm_door(z, 1.), width=0.1, isotropic=False)
    ]


LIKELIHOODS = create_likelihoods()


@pytest.mark.parametrize("func,likelihood", [
    (np.abs, AbsLikelihood(y=None)),
    (np.sign, SgnLikelihood(y=None)),
    (relu, ReluLikelihood(y=None)),
    (hard_tanh, HardTanhLikelihood(y=None)),
    (hard_sigm, HardSigmoidLikelihood(y=None)),
    (lambda z: leaky_relu(z, slope=0.1), LeakyReluLikelihood(y=None, slope=0.1)),
    (lambda z: symm_door(z, width=2.), SymmetricDoorLikelihood(y=None, width=2.))
])
def test_likelihood_sample(func, likelihood):
    z = np.linspace(-3, 3, 100)
    assert_allclose(func(z), likelihood.sample(z))


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_separable_likelihood_vectorization(likelihood):
    assert not likelihood.isotropic
    N = np.prod(likelihood.size)
    az = np.linspace(1, 2, N)
    az = az.reshape(likelihood.size)
    bz = np.linspace(-2, 2, N)
    bz = bz.reshape(likelihood.size)
    rz, vz = likelihood.compute_backward_posterior(az, bz, likelihood.y)
    assert rz.shape == bz.shape
    assert vz.shape == bz.shape
    # check rz vectorization
    rz_ = np.array([
        likelihood.scalar_backward_mean(a, b, y)
        for a, b, y in zip(az, bz, likelihood.y)
    ])
    rz_ = rz_.reshape(likelihood.size)
    assert_allclose(rz, rz_)
    # check vz vectorization
    vz_ = np.array([
        likelihood.scalar_backward_variance(a, b, y)
        for a, b, y in zip(az, bz, likelihood.y)
    ])
    vz_ = vz_.reshape(likelihood.size)
    assert_allclose(vz, vz_)
    # check A vectorization
    A = likelihood.compute_log_partition(az, bz, likelihood.y)
    A_ = np.mean([
        likelihood.scalar_log_partition(a, b, y)
        for a, b, y in zip(az, bz, likelihood.y)
    ])
    assert A == A_


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_grad_EP_scalar(likelihood):
    df = check_likelihood_grad_EP_scalar(likelihood)
    assert_allclose(df["rz"], df["grad_bz_A1"], rtol=0, atol=EPSILON)
    assert_allclose(df["vz"], df["grad_bz_A2"], rtol=0, atol=EPSILON)
    assert_allclose(df["tz"], -2*df["grad_az_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_grad_EP_diagonal(likelihood):
    assert not likelihood.isotropic
    df = check_likelihood_grad_EP_diagonal(likelihood)
    assert_allclose(df["rz"], df["grad_bz_A1"], rtol=0, atol=EPSILON)
    assert_allclose(df["vz"], df["grad_bz_A2"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_grad_FG(likelihood):
    df = check_likelihood_grad_FG(likelihood)
    assert_allclose(df["tz"], -2*df["grad_tz_hat_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_BO_limit(likelihood):
    df = check_likelihood_BO_limit(likelihood)
    assert_allclose(df["A_BO"], df["A_RS"], rtol=0, atol=EPSILON)
    assert_allclose(df["vz_BO"], df["vz_RS"], rtol=0, atol=EPSILON)
    assert_allclose(df["mz_BO"], df["mz_RS"], rtol=0, atol=EPSILON)
    assert_allclose(df["mz_BO"], df["qz_RS"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_BN_limit(likelihood):
    df = check_likelihood_BN_limit(likelihood)
    assert_allclose(df["A_FG"], df["A_BN"], rtol=0, atol=EPSILON)
    assert_allclose(df["vz_FG"], df["vz_BN"], rtol=0, atol=EPSILON)
    assert_allclose(df["tz_FG"], df["tz_BN"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_likelihood_grad_BO(likelihood):
    df = check_likelihood_grad_BO(likelihood)
    assert_allclose(df["mz"], 2*df["grad_mz_hat_A"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("likelihood", LIKELIHOODS + [
    ModulusLikelihood(y=None)
])
def test_likelihood_grad_BO_BN(likelihood):
    df = check_likelihood_grad_BO_BN(likelihood)
    assert_allclose(df["mz"], 2*df["grad_az_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["vz"], 2*df["grad_az_I"], rtol=0, atol=EPSILON)


@pytest.mark.parametrize("teacher,student", [
    (likelihood, AbsLikelihood(y=None)) for likelihood in LIKELIHOODS
])
def test_likelihood_grad_RS(teacher, student):
    df = check_likelihood_grad_RS(teacher, student)
    assert_allclose(df["mz"], df["grad_mz_hat_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["qz"], -2*df["grad_qz_hat_A"], rtol=0, atol=EPSILON)
    assert_allclose(df["tz"], -2*df["grad_tz_hat_A"], rtol=0, atol=EPSILON)

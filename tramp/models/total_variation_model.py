import numpy as np
from ..variables import SISOVariable as V, SIMOVariable, MILeafVariable
from ..channels import (
    LinearChannel, GaussianChannel, GradientChannel, ReshapeChannel
)
from ..priors import GaussianPrior, GaussBernoulliPrior, MAP_L21NormPrior
from ..likelihoods import GaussianLikelihood, SgnLikelihood


def sparse_gradient_block(x_shape, prior_var, grad_rho):
    d = len(x_shape)
    grad_shape = (d,) + x_shape
    N = np.product(x_shape)
    block = (
        GaussianPrior(size=x_shape, var=prior_var) @
        SIMOVariable(id="x", n_next=2) @ ((
                GradientChannel(shape=x_shape) +
                GaussBernoulliPrior(size=grad_shape, rho=grad_rho)
            ) @ MILeafVariable(id="x'", n_prev=2)
        )
    ) @ ReshapeChannel(prev_shape=x_shape, next_shape=N)
    return block


def tv_block(x_shape, prior_var, grad_scale):
    d = len(x_shape)
    grad_shape = (d,) + x_shape
    N = np.product(x_shape)
    block = (
        GaussianPrior(size=x_shape, var=prior_var) @
        SIMOVariable(id="x", n_next=2) @ ((
                GradientChannel(shape=x_shape) +
                MAP_L21NormPrior(size=grad_shape, scale=grad_scale, axis=0)
            ) @ MILeafVariable(id="x'", n_prev=2)
        )
    ) @ ReshapeChannel(prev_shape=x_shape, next_shape=N)
    return block


def regression_block(A, y, noise_var):
    block = (
        LinearChannel(A, name="A") @ V(id="z") @
        GaussianLikelihood(y, var=noise_var)
    )
    return block


def classification_block(A, y, noise_var):
    block = (
        LinearChannel(A, name="A") @ V(id="z") @
        GaussianChannel(var=noise_var) @ V(id="a") @ SgnLikelihood(y)
    )
    return block


def sparse_gradient_regression(A, y, x_shape, grad_rho, noise_var, prior_var):
    sparse_grad = sparse_gradient_block(x_shape, prior_var, grad_rho)
    reg = regression_block(A, y, noise_var)
    model = (sparse_grad @ V(id="r") @ reg).to_model()
    return model


def sparse_gradient_classification(A, y, x_shape, grad_rho, noise_var, prior_var):
    sparse_grad = sparse_gradient_block(x_shape, prior_var, grad_rho)
    clf = classification_block(A, y, noise_var)
    model = (sparse_grad @ V(id="r") @ clf).to_model()
    return model


def tv_regression(A, y, x_shape, grad_scale, noise_var, prior_var):
    tv = tv_block(x_shape, prior_var, grad_scale)
    reg = regression_block(A, y, noise_var)
    model = (tv @ V(id="r") @ reg).to_model()
    return model


def tv_classification(A, y, x_shape, grad_scale, noise_var, prior_var):
    tv = tv_block(x_shape, prior_var, grad_scale)
    clf = classification_block(A, y, noise_var)
    model = (tv @ V(id="r") @ clf).to_model()
    return model

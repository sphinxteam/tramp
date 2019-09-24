import numpy as np
from .base_model import Model
from ..variables import SISOVariable, SIMOVariable, MILeafVariable
from ..channels import LinearChannel, GaussianChannel, GradientChannel
from ..priors import GaussianPrior, GaussBernouilliPrior, MAP_L21NormPrior
from ..likelihoods import GaussianLikelihood, SgnLikelihood


class SparseGradientRegression(Model):
    def __init__(self, A, y, x_shape, rho_grad, var_noise, var_prior):
        M, N = A.shape
        if len(y) != M:
            raise ValueError(f"Bad shape y: {len(y)} A: {A.shape}")
        if np.product(x_shape) != N:
            raise ValueError(f"Bad shape x: {x_shape} A: {A.shape}")
        d = len(x_shape)
        ravel = (d > 1)
        grad_shape = (d,) + x_shape
        self.x_shape = x_shape
        self.rho_grad = rho_grad
        self.var_noise = var_noise
        self.var_prior = var_prior
        self.repr_init()
        model_dag = (
            GaussianPrior(size=x_shape, var=var_prior) @
            SIMOVariable(id="x", n_next=2) @ (
                LinearChannel(A, ravel=ravel, W_name="A") @
                SISOVariable(id="z") @
                GaussianLikelihood(y, var=var_noise) + (
                    GradientChannel(shape=x_shape) +
                    GaussBernouilliPrior(size=grad_shape, rho=rho_grad)
                ) @
                MILeafVariable(id="x'", n_prev=2)
            )
        ).to_model_dag()
        Model.__init__(self, model_dag)


class SparseGradientClassification(Model):
    def __init__(self, A, y, x_shape, rho_grad, var_noise, var_prior):
        M, N = A.shape
        if len(y) != M:
            raise ValueError(f"Bad shape y: {len(y)} A: {A.shape}")
        if np.product(x_shape) != N:
            raise ValueError(f"Bad shape x: {x_shape} A: {A.shape}")
        d = len(x_shape)
        ravel = (d > 1)
        grad_shape = (d,) + x_shape
        self.x_shape = x_shape
        self.rho_grad = rho_grad
        self.var_noise = var_noise
        self.var_prior = var_prior
        self.repr_init()
        model_dag = (
            GaussianPrior(size=x_shape, var=var_prior) @
            SIMOVariable(id="x", n_next=2) @ (
                LinearChannel(A, ravel=ravel, W_name="A") @
                SISOVariable(id="z") @
                GaussianChannel(var=var_noise) @
                SISOVariable(id="a") @
                SgnLikelihood(y) + (
                    GradientChannel(shape=x_shape) +
                    GaussBernouilliPrior(size=grad_shape, rho=rho_grad)
                ) @
                MILeafVariable(id="x'", n_prev=2)
            )
        ).to_model_dag()
        Model.__init__(self, model_dag)


class TVRegression(Model):
    def __init__(self, A, y, x_shape, scale_grad, var_noise, var_prior):
        M, N = A.shape
        if len(y) != M:
            raise ValueError(f"Bad shape y: {len(y)} A: {A.shape}")
        if np.product(x_shape) != N:
            raise ValueError(f"Bad shape x: {x_shape} A: {A.shape}")
        d = len(x_shape)
        ravel = (d > 1)
        grad_shape = (d,) + x_shape
        self.x_shape = x_shape
        self.scale_grad = scale_grad
        self.var_noise = var_noise
        self.var_prior = var_prior
        self.repr_init()
        model_dag = (
            GaussianPrior(size=x_shape, var=var_prior) @
            SIMOVariable(id="x", n_next=2) @ (
                LinearChannel(A, ravel=ravel, W_name="A") @
                SISOVariable(id="z") @
                GaussianLikelihood(y, var=var_noise) + (
                    GradientChannel(shape=x_shape) +
                    MAP_L21NormPrior(size=grad_shape, scale=scale_grad, axis=0)
                ) @
                MILeafVariable(id="x'", n_prev=2)
            )
        ).to_model_dag()
        Model.__init__(self, model_dag)


class TVClassification(Model):
    def __init__(self, A, y, x_shape, scale_grad, var_noise, var_prior):
        M, N = A.shape
        if len(y) != M:
            raise ValueError(f"Bad shape y: {len(y)} A: {A.shape}")
        if np.product(x_shape) != N:
            raise ValueError(f"Bad shape x: {x_shape} A: {A.shape}")
        d = len(x_shape)
        ravel = (d > 1)
        grad_shape = (d,) + x_shape
        self.x_shape = x_shape
        self.scale_grad = scale_grad
        self.var_noise = var_noise
        self.var_prior = var_prior
        self.repr_init()
        model_dag = (
            GaussianPrior(size=x_shape, var=var_prior) @
            SIMOVariable(id="x", n_next=2) @ (
                LinearChannel(A, ravel=ravel, W_name="A") @
                SISOVariable(id="z") @
                GaussianChannel(var=var_noise) @
                SISOVariable(id="a") @
                SgnLikelihood(y) + (
                    GradientChannel(shape=x_shape) +
                    MAP_L21NormPrior(size=grad_shape, scale=scale_grad, axis=0)
                ) @
                MILeafVariable(id="x'", n_prev=2)
            )
        ).to_model_dag()
        Model.__init__(self, model_dag)

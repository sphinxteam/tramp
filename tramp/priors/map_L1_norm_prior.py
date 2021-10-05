"""Implements the MAP_L1NormPrior class."""
import numpy as np
from .base_prior import Prior


def l1_norm(x):
    x_abs = np.abs(x)
    return x_abs.sum()


def soft_threshold(x, gamma):
    x_abs = np.abs(x)
    return np.maximum(0, 1 - gamma / x_abs) * x


def v_soft_threshold(x, gamma):
    return np.abs(x) > gamma


class MAP_L1NormPrior(Prior):
    r"""MAP prior associated to the $\Vert . \Vert_1$ penalty.

    The corresponding factor is given by $f(x) = e^{-\gamma \Vert x \Vert_1}$
    where $\gamma$ is the regularization parameter.

    Parameters
    ----------
    size : int or tuple of int
        Shape of x
    gamma : float
        Regularization parameter $\gamma$
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, gamma=1, isotropic=True):
        self.size = size
        self.gamma = gamma
        self.isotropic = isotropic
        self.repr_init()

    def sample(self):
        X = np.random.laplace(size=self.size, scale=1/self.gamma)
        return X

    def math(self):
        return r"$\Vert . \Vert_1$"

    def second_moment(self):
        raise NotImplementedError

    def forward_second_moment_FG(self, tx_hat):
        raise NotImplementedError

    def scalar_forward_mean(self, ax, bx):
        return (1 / ax) * soft_threshold(bx, self.gamma)

    def scalar_forward_variance(self, ax, bx):
        return (1 / ax) * v_soft_threshold(bx, self.gamma)

    def scalar_log_partition(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        A =  bx*rx - 0.5*ax*(rx**2) - self.gamma*np.abs(rx)
        return A

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        vx = (1 / ax) * v_soft_threshold(bx, self.gamma)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        rx = (1 / ax) * soft_threshold(bx, self.gamma)
        A =  bx*rx - 0.5*ax*(rx**2) - self.gamma*np.abs(rx)
        return A.mean()

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError

"""Implements the MAP_L21NormPrior class."""
import numpy as np
from .base_prior import Prior
import warnings


def l21_norm(x, axis):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=False)
    return x_norm.sum()


def group_soft_threshold(x, gamma, axis):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)  # broadcast against x
    return np.maximum(0, 1 - gamma / x_norm) * x


def v_group_soft_threshold(x, gamma, axis):
    x_norm = np.linalg.norm(x, axis=axis, keepdims=False)
    v = (x_norm > gamma) * (1 + (x**2 / x_norm**2 - 1) * gamma / x_norm)
    return v


class MAP_L21NormPrior(Prior):
    r"""MAP prior associated to the $\Vert . \Vert_{2,1}$ penalty.

    The corresponding factor is given by $f(x)=e^{-\gamma \Vert x \Vert_{2,1}}$
    where $\gamma$ is the regularization parameter.

    Parameters
    ----------
    size : tuple of int
        Shape of x
    gamma : float
        Regularization parameter $\gamma$
    axis : int
        Axis over which the $\Vert . \Vert_2$ norm is taken
    isotropic : bool
        Using isotropic or diagonal beliefs
    """

    def __init__(self, size, gamma=1, axis=0, isotropic=True):
        assert type(size)==tuple and len(size)>1, "size must be a tuple of length > 1"
        self.size = size
        self.gamma = gamma
        self.axis = axis
        self.isotropic = isotropic
        self.repr_init()
        self.N = np.prod(size)
        self.d = size[axis]

    def sample(self):
        warnings.warn(
            "MAP_L21NormPrior.sample not implemented "
            "return zero array as a placeholder"
        )
        return np.zeros(self.size)

    def math(self):
        return r"$\Vert . \Vert_{2,1}$"

    def second_moment(self):
        raise NotImplementedError

    def forward_second_moment_FG(self, tx_hat):
        raise NotImplementedError

    def compute_forward_posterior(self, ax, bx):
        rx = (1 / ax) * group_soft_threshold(bx, self.gamma, self.axis)
        vx = (1 / ax) * v_group_soft_threshold(bx, self.gamma, self.axis)
        if self.isotropic:
            vx = vx.mean()
        return rx, vx

    def compute_log_partition(self, ax, bx):
        rx = (1 / ax) * group_soft_threshold(bx, self.gamma, self.axis)
        A_sum = np.sum(bx*rx - 0.5*ax*(rx**2)) - self.gamma*l21_norm(rx, self.axis)
        return A_sum / self.N

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        raise NotImplementedError

    def measure(self, f):
        raise NotImplementedError

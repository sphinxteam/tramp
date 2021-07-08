"""Implements the CommitteeBinaryPrior class."""
import numpy as np
from scipy.special import logsumexp, softmax
from .base_prior import Prior
from ..utils.integration import gaussian_measure_full
from ..beliefs import binary


def create_spins(K):
    "Create spins configurations"
    x = [list(np.binary_repr(i, width=K)) for i in range(2**K)]
    x = 2*np.array(x, dtype=int) - 1  # shape (2**K, K)
    return x


def compute_px(p_pos, x):
    "Compute spins probabilties"
    n_pos = (x == +1).sum(axis=1)  # shape (2**K,)
    n_neg = (x == -1).sum(axis=1)  # shape (2**K,)
    p_neg = 1 - p_pos
    px = (p_pos**n_pos)*(p_neg**n_neg)  # shape (2**K,)
    return px


def compute_C(x):
    "Compute spins covariance matrix C = (x-x')(x-x')^T"
    n = np.newaxis
    # entries c,d,k,l
    x_ck = x[:, n, :, n]
    x_dk = x[n, :, :, n]
    x_cl = x[:, n, n, :]
    x_dl = x[n, :, n, :]
    C = (x_ck - x_dk)*(x_cl - x_dl)  # shape (2**K, 2**K, K, K)
    return C


def compute_Ax_vector(x, a, b):
    """Compute Ax = - 1/2 x.ax + b.x for b of shape (N, K)

    Parameters
    ----------
    x : array of shape (2**K, K)
    a : arrary of shape (K, K)
    b : array of shape (N, K)

    Returns
    -------
    array of shape (N, 2**K)
        Ax_ic =   - 1/2 sum_k x_ck a_kl x_cl + sum_k b_ik x_ck
    """
    xax = np.einsum("ck,ck -> c", x @ a, x)
    bx = np.einsum("ik,ck -> ic", b, x)
    Ax = -0.5*xax + bx
    return Ax


def compute_Ax_scalar(x, a, b):
    """Compute Ax = - 1/2 x.ax + b.x for b of shape (K,)

    Parameters
    ----------
    x : array of shape (2**K, K)
    a : arrary of shape (K, K)
    b : array of shape (K,)

    Returns
    -------
    array of shape (2**K,)
        Ax_c =   - 1/2 sum_kl x_ck a_kl x_cl + sum_k b_k x_ck
    """
    xax = np.einsum("ck,ck -> c", x @ a, x)
    bx = x @ b
    Ax = -0.5*xax + bx
    return Ax


def compute_V_vector(p, C):
    """Compute V = 1/ N sum_xx' p_x p_x' C_xx'

    Parameters
    ----------
    p : array of shape (N, 2**K)
    C : array of shape (2**K, 2**K, K, K)
        spin configurations obtained by x=create_spins(K); C=compute_C(x)

    Returns
    -------
    array of shape (K, K)
        V_kl =  1/N sum_i sum_cd p_ic p_id C_cdkl
    """
    N = p.shape[0]
    w = np.einsum("ic,id -> cd", p, p) / N
    V = np.einsum("cd,cdkl -> kl", w, C)
    return V


def compute_V_scalar(p, C):
    """Compute V = sum_xx' p_x p_x' C_xx'

    Parameters
    ----------
    p : array of shape (2**K,)
    C : array of shape (2**K, 2**K, K, K)
        spin configurations obtained by x=create_spins(K); C=compute_C(x)

    Returns
    -------
    array of shape (K, K)
        V_kl =  sum_cd p_c p_d C_cdkl
    """
    N = p.shape[0]
    w = np.einsum("c,d -> cd", p, p)
    V = np.einsum("cd,cdkl -> kl", w, C)
    return V


class CommitteeBinaryPrior(Prior):
    r"""Committee Binary prior :math:`p(x) = p_+ \delta_+(x) + p_- \delta_-(x)`

    Parameters
    ----------
    N : int
        Shape of x is (N, K)
    K : int
        Number of experts in the committee
    p_pos : float in (0,1)
        Parameter :math:`p_+` of the binary prior
    """

    def __init__(self, N, K, p_pos=0.5):
        self.N = N
        self.K = K
        self.p_pos = p_pos
        self.repr_init()
        self.size = (N, K)
        self.x = create_spins(K)  # shape (2**K, K)
        self.C = compute_C(self.x)  # shape (2**K, 2**k, K, K)
        self.px = compute_px(p_pos, self.x)  # shape (2**K,)
        self.p_neg = 1 - p_pos
        # natural parameters
        self.b = 0.5*np.log(self.p_pos / self.p_neg)

    def sample(self):
        p = [self.p_neg, self.p_pos]
        X = np.random.choice([-1, +1], size=self.size, replace=True, p=p)
        return X

    def math(self):
        return r"$p_\pm$"

    def second_moment(self):
        return 1.

    def scalar_forward_mean(self, ax, bx):
        b = bx + self.b
        Ax = compute_Ax_scalar(self.x, ax, b)  # shape (2**K,)
        prob = softmax(Ax)  # shape (2**K,)
        rx = prob @ self.x  # shape (K,)
        return rx

    def scalar_forward_variance(self, ax, bx):
        b = bx + self.b
        Ax = compute_Ax_scalar(self.x, ax, b)  # shape (2**K,)
        prob = softmax(Ax)  # shape (2**K,)
        vx = compute_V_scalar(prob, self.C)  # shape (K, K)
        return vx

    def scalar_log_partition(self, ax, bx):
        b = bx + self.b
        Ax = compute_Ax_scalar(self.x, ax, b)  # shape (2**K,)
        A = logsumexp(Ax)/self.K - binary.A(self.b)
        return A

    def compute_forward_posterior(self, ax, bx):
        b = bx + self.b
        Ax = compute_Ax_vector(self.x, ax, b)  # shape (N, 2**K)
        prob = softmax(Ax, axis=1)  # shape (N, 2**K)
        rx = prob @ self.x  # shape (N, K)
        vx = compute_V_vector(prob, self.C)  # shape (K, K)
        return rx, vx

    def compute_log_partition(self, ax, bx):
        b = bx + self.b
        Ax = compute_Ax_vector(self.x, ax, b)  # shape (N, 2**K)
        A = logsumexp(Ax, axis=1).mean() - binary.A(self.b)
        return A

    def b_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def bx_measure(self, mx_hat, qx_hat, tx0_hat, f):
        raise NotImplementedError

    def beliefs_measure(self, ax, f):
        mu = 0
        for x, px in zip(self.x, self.px):
            mu += px * gaussian_measure_full(ax @ x, ax, f)
        return mu

    def measure(self, f):
        return self.p_pos * f(+1) + self.p_neg * f(-1)

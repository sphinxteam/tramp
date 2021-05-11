import numpy as np
from scipy.special import logsumexp, softmax
from .base_prior import Prior
from ..utils.integration import gaussian_measure_full


def create_spins(K):
    "Create spins configurations"
    x = [list(np.binary_repr(i, width=K)) for i in range(2**K)]
    x = 2*np.array(x, dtype=int) - 1 # shape (2**K, K)
    return x

def compute_px(p_pos, x):
    "Compute spins probabilties"
    n_pos = (x==+1).sum(axis=1) # shape (2**K,)
    n_neg = (x==-1).sum(axis=1) # shape (2**K,)
    p_neg = 1 - p_pos
    px = (p_pos**n_pos)*(p_neg**n_neg) # shape (2**K,)
    return px


def compute_C(x):
    "Compute spins covariance matrix C = (x-x')(x-x')^T"
    n = np.newaxis
    #entries c,d,k,l
    x_ck = x[:,n,:,n]
    x_dk = x[n,:,:,n]
    x_cl = x[:,n,n,:]
    x_dl = x[n,:,n,:]
    C = (x_ck - x_dk)*(x_cl - x_dl) # shape (2**K, 2**K, K, K)
    return C


def compute_A_vector(x, a, b):
    """Compute A = - 1/2 x.ax + b.x

    Parameters
    ----------
    x : array of shape (2**K, K)
    a : arrary of shape (K, K)
    b : array of shape (N, K)

    Returns
    -------
    array of shape (N, 2**K)
        A_ic =   - 1/2 sum_k x_ck a_kl x_cl + sum_k b_ik x_ck
    """
    xax = np.einsum("ck,ck -> c", x @ a, x)
    bx = np.einsum("ik,ck -> ic", b, x)
    A = -0.5*xax + bx
    return A


def compute_A_scalar(x, a, b):
    """Compute A = - 1/2 x.ax + b.x

    Parameters
    ----------
    x : array of shape (2**K, K)
    a : arrary of shape (K, K)
    b : array of shape (K,)

    Returns
    -------
    array of shape (2**K,)
        A_c =   - 1/2 sum_kl x_ck a_kl x_cl + sum_k b_k x_ck
    """
    xax = np.einsum("ck,ck -> c", x @ a, x)
    bx = x @ b
    A = -0.5*xax + bx
    return A


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
    def __init__(self, N, K, p_pos=0.5):
        self.N = N
        self.K = K
        self.p_pos = p_pos
        self.repr_init()
        self.size = (N, K)
        self.x = create_spins(K) # shape (2**K, K)
        self.C = compute_C(self.x) # shape (2**K, 2**k, K, K)
        self.px = compute_px(p_pos, self.x) # shape (2**K,)
        self.p_neg = 1 - p_pos
        self.log_odds = np.log(self.p_pos / self.p_neg)
        self.b = 0.5*self.log_odds
        self.A = self.K * np.log(2*np.cosh(self.b))

    def sample(self):
        p = [self.p_neg, self.p_pos]
        X = np.random.choice([-1, +1], size=self.size, replace=True, p=p)
        return X

    def math(self):
        return r"$p_\pm$"

    def second_moment(self):
        return 1.

    def compute_forward_posterior(self, ax, bx):
        b = bx + self.b
        A = compute_A_vector(self.x, ax, bx) # shape (N, 2**K)
        prob = softmax(A, axis=1) # shape (N, 2**K)
        rx = prob @ self.x # shape (N, K)
        vx = compute_V_vector(prob, self.C) # shape (K, K)
        return rx, vx

    def beliefs_measure(self, ax, f):
        mu = 0
        for x, px in zip(self.x, self.px):
            mu += px * gaussian_measure_full(ax @ x, ax, f)
        return mu

    def measure(self, f):
        return self.p_pos * f(+1) + self.p_neg * f(-1)

    def compute_log_partition(self, ax, bx):
        b = bx + self.b
        A = compute_A_vector(self.x, ax, bx) # shape (N, 2**K)
        logZ = logsumexp(A, axis=1).sum() - self.A
        return logZ

    def scalar_log_partition(self, ax, bx):
        b = bx + self.b
        A = compute_A_scalar(self.x, ax, bx) # shape (2**K,)
        logZ = logsumexp(A) - self.A
        return logZ

    def scalar_variance(self, ax, bx):
        b = bx + self.b
        A = compute_A_scalar(self.x, ax, bx) # shape (2**K,)
        prob = softmax(A) # shape (2**K,)
        vx = compute_V_scalar(prob, self.C) # shape (K, K)
        return vx

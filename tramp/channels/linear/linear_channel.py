import numpy as np
from ..base_channel import Channel
import logging
logger = logging.getLogger(__name__)



def svd(X):
    "Compute SVD of X = U S V.T"
    U, s, VT = np.linalg.svd(X, full_matrices=True)
    V = VT.T
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    svd_X = (U, S, V)
    return svd_X


class LinearChannel(Channel):
    """Linear channel x = W z.

    Parameters
    ----------
    - W: array of shape (Nx, Nz)
    - precompute_svd: bool
        if True precompute SVD of W = U S V.T
    - name: str
        name of weight matrix W for display
    """

    def __init__(self, W, precompute_svd=True, name="W"):
        self.name = name
        self.Nx = W.shape[0]
        self.Nz = W.shape[1]
        self.precompute_svd = precompute_svd
        self.repr_init()
        self.W = W
        self.rank = np.linalg.matrix_rank(W)
        self.alpha = self.Nx / self.Nz
        if precompute_svd:
            self.U, self.S, self.V = svd(W)
            self.spectrum = np.diag(self.S.T @ self.S)
        else:
            self.C = W.T @ W
            self.spectrum = np.linalg.eigvalsh(self.C)
        assert self.spectrum.shape == (self.Nz,)
        self.singular = self.spectrum[:self.rank]

    def sample(self, Z):
        X = self.W @ Z
        return X

    def math(self):
        return r"$" + self.name + "$"

    def second_moment(self, tau_z):
        return tau_z * self.spectrum.sum() / self.Nx

    def compute_n_eff(self, az, ax):
        "Effective number of parameters = overlap in z"
        if ax == 0:
            logger.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logger.info(f"az/ax=0 in {self} compute_n_eff")
            return self.rank / self.Nz
        n_eff_trace = np.sum(self.singular / (az / ax + self.singular))
        return n_eff_trace / self.Nz

    def compute_backward_mean(self, az, bz, ax, bx):
        # estimate z from x = Wz
        if self.precompute_svd:
            bx_svd = self.U.T @ bx
            bz_svd = self.V.T @ bz
            resolvent = 1 / (az + ax * self.spectrum)
            if bz.ndim > 1:
                resolvent = resolvent[:, np.newaxis]
            rz_svd = resolvent * (bz_svd + self.S.T @ bx_svd)
            rz = self.V @ rz_svd
        else:
            a = az * np.identity(self.Nz) + ax * self.C
            b = (bz + self.W.T @ bx)
            rz = np.linalg.solve(a, b)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        # estimate x from x = Wz we have rx = W rz
        rz = self.compute_backward_mean(az, bz, ax, bx)
        rx = self.W @ rz
        return rx

    def compute_backward_variance(self, az, ax):
        if az==0:
            logger.info(f"az=0 in {self} compute_backward_variance, clipping to 1e-11")
        az = np.maximum(1e-11, az)
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            s_mean = np.mean(self.singular)
            return s_mean * self.rank / (self.Nx * az)
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / (self.alpha * ax)
        return vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        # estimate z from x = Wz
        rz = self.compute_backward_mean(az, bz, ax, bx)
        vz = self.compute_backward_variance(az, ax)
        return rz, vz

    def compute_forward_posterior(self, az, bz, ax, bx):
        # estimate x from x = Wz
        rx = self.compute_forward_mean(az, bz, ax, bx)
        vx = self.compute_forward_variance(az, ax)
        return rx, vx

    def compute_backward_error(self, az, ax, tau_z):
        vz = self.compute_backward_variance(az, ax)
        return vz

    def compute_forward_error(self, az, ax, tau_z):
        vx = self.compute_forward_variance(az, ax)
        return vx

    def compute_log_partition(self, az, bz, ax, bx):
        rz = self.compute_backward_mean(az, bz, ax, bx)
        b = bz + self.W.T @ bx
        a = az + ax * self.spectrum
        logZ = 0.5 * np.sum(b * rz) + 0.5 * np.sum(np.log(2 * np.pi / a))
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        I = 0.5*np.log((az + ax * self.spectrum)*tau_z)
        I = I.mean()
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + self.alpha*ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

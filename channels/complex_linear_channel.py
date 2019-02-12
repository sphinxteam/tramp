import numpy as np
from ..base import Channel
import logging


def complex_svd(X):
    "Compute SVD of X = U S V.H"
    U, s, VH = np.linalg.svd(X, full_matrices=True)
    V = VH.H
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    svd_X = (U, S, V)
    return svd_X


def complex2array(z):
    """Transform complex z into real array Z where:
    - Z[0] = z.real
    - Z[1] = z.imag
    """
    Z_shape = (2,) + z.shape
    Z = np.zeros(Z_shape)
    Z[0] = z.real
    Z[1] = z.imag
    return Z


def array2complex(Z):
    """Transform real array Z into complex z where:
    - z.real = Z[0]
    - z.imag = Z[1]
    """
    if Z.shape[0] != 2:
        raise ValueError(
            "first axis of Z must be of length 2"
            "where Z[0] = Z.real and Z[1] = Z.imag"
        )
    z = Z[0] + 1j * Z[1]
    return z

class ComplexLinearChannel(Channel):
    """Complex linear channel x = W z.

    Parameters
    ----------
    - W: real or complex array of shape (Nx, Nz)
    - precompute_svd: bool
        if True precompute SVD of W = U S V.H
    - ravel: bool
        if True  x = W @ z.ravel()
        if False x = W @ z
    - W_name: str
        name of weight matrix W for display

    Notes
    -----
    For message passing it is more convenient to represent a complex array x
    as a real array X where X[0] = x.real and X[1] = x.imag

    In particular:
    - input  of sample(): Z array of shape (2, z.shape)
    - output of sample(): X array of shape (2, x.shape)
    - message bz, posterior rz: real arrays of shape (2, z.shape)
    - message bx, posterior rx: real arrays of shape (2, x.shape)
    """

    def __init__(self, W, ravel=False, precompute_svd=True, W_name="W"):
        W = np.matrix(W)
        self.W_name = W_name
        self.Nx = W.shape[0]
        # TODO check Nz when ravel = False and z matrix
        self.Nz = W.shape[1]
        self.ravel = ravel
        self.precompute_svd = precompute_svd
        self.repr_init()
        self.W = W
        self.rank = np.linalg.matrix_rank(W)
        self.alpha = self.Nx / self.Nz
        if precompute_svd:
            self.U, self.S, self.V = complex_svd(W)
            self.spectrum = np.diag(self.S.H @ self.S)
        else:
            self.C = W.H @ W
            self.spectrum = np.linalg.eigvalsh(self.C)
        assert self.spectrum.shape == (self.Nz,)
        self.singular = self.spectrum[:self.rank]

    def sample(self, Z):
        "We assume Z[0] = Z.real and Z[1] = Z.imag"
        Z = array2complex(Z)
        if self.ravel:
            Z = Z.ravel()
        X = self.W @ Z
        X = complex2array(X)
        assert X.shape == (2, self.Nx)
        return X

    def math(self):
        return r"$" + self.W_name + "$"

    def second_moment(self, tau):
        return tau * self.spectrum.sum() / self.Nx

    def compute_n_eff(self, az, ax):
        "Effective number of parameters = overlap in z"
        if ax == 0:
            logging.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logging.info(f"az/ax=0 in {self} compute_n_eff")
            return self.rank / self.Nz
        n_eff_trace = np.sum(self.singular / (az / ax + self.singular))
        return n_eff_trace / self.Nz

    def compute_backward_mean(self, az, bz, ax, bx):
        # estimate z from x = Wz
        bz = array2complex(bz)
        bx = array2complex(bx)
        if self.ravel:
            z_shape = bz.shape
            bz = bz.ravel()
        if self.precompute_svd:
            bx_svd = self.U.H @ bx
            bz_svd = self.V.H @ bz
            resolvent = 1 / (az + ax * self.spectrum)
            rz_svd = resolvent * (bz_svd + self.S.H @ bx_svd)
            rz = self.V @ rz_svd
        else:
            a = az * np.identity(self.Nz) + ax * self.C
            b = (bz + self.W.H @ bx)
            rz = np.linalg.solve(a, b)
        if self.ravel:
            rz = rz.reshape(z_shape)
        rz = complex2array(rz)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        # estimate x from x = Wz we have rx = W rz
        rz = self.compute_backward_mean(az, bz, ax, bx)
        rz = array2complex(rz)
        if self.ravel:
            rz = rz.ravel()
        rx = self.W @ rz
        rx = complex2array(rx)
        return rx

    def compute_backward_variance(self, az, ax):
        if az == 0:
            logging.info(f"az=0 in {self} compute_backward_variance")
            i_mean = np.mean(1 / self.singular)
            return np.inf if self.rank < self.Nz else i_mean / ax
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            logging.info(f"ax=0 in {self} compute_forward_variance")
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

    def compute_backward_error(self, az, ax, tau):
        vz = self.compute_backward_variance(az, ax)
        return vz

    def compute_forward_error(self, az, ax, tau):
        vx = self.compute_forward_variance(az, ax)
        return vx

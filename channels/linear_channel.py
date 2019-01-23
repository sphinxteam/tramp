import numpy as np
from ..base import Channel


def svd(X):
    "Compute SVD of X = U S V.T"
    U, s, VT = np.linalg.svd(X, full_matrices=True)
    V = VT.T
    S = np.zeros((U.shape[0], V.shape[0]))
    S[:len(s), :len(s)] = np.diag(s)
    svd_X = (U, S, V)
    return svd_X


class LinearChannel(Channel):
    def __init__(self, W, precompute_svd = True):
        self.Nx = W.shape[0]
        self.Nz = W.shape[1]
        self.precompute_svd = True
        self.repr_init()
        self.W = W
        if precompute_svd:
            self.U, S, self.V = svd(W)
            self.S = S
            self.spectrum = np.diag(S.T @ S)
            assert self.spectrum.shape == (self.Nz,)
        else:
            self.C = W.T @ W
            self.spectrum = np.linalg.eigvalsh(self.C)
            assert self.spectrum.shape == (self.Nz,)

    def sample(self, Z):
        X = self.W @ Z
        return X

    def math(self):
        return r"$W$"

    def second_moment(self, tau):
        return tau * self.spectrum.sum() / self.Nx

    def backward_posterior(self, message):
        # estimate z from x = Wz
        az, bz, ax, bx = self._parse_message_ab(message)
        resolvent = 1 / (az + ax * self.spectrum)
        if self.precompute_svd:
            bx_svd = self.U.T @ bx
            bz_svd = self.V.T @ bz
            rz_svd = resolvent * (bz_svd + self.S.T @ bx_svd)
            rz = self.V @ rz_svd
        else:
            a = az * np.identity(self.Nz) + ax * self.C
            b = (bz + self.W.T @ bx)
            rz = np.linalg.solve(a, b)
        vz = resolvent.mean()
        return [(rz, vz)]

    def forward_posterior(self, message):
        # estimate x from x = Wz
        az, bz, ax, bx = self._parse_message_ab(message)
        rz, vz = self.backward_posterior(message)[0]
        # rx = W rz and Nx ax vx + Nz az vz = Nz
        rx = self.W @ rz
        vx = (self.Nz / self.Nx) * (1 - az * vz) / ax
        return [(rx, vx)]

    def backward_error(self, message):
        az, ax = self._parse_message_a(message)
        resolvent = 1 / (az + ax * self.spectrum)
        vz = resolvent.mean()
        return [vz]

    def forward_error(self, message):
        az, ax = self._parse_message_a(message)
        resolvent = 1 / (az + ax * self.spectrum)
        vz = resolvent.mean()
        vx = (self.Nz / self.Nx) * (1 - az * vz) / ax
        return [vx]

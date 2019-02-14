import numpy as np
from numpy.fft import fftn, ifftn
from ..base import Channel
from .complex_linear_channel import complex2array, array2complex


class DFTChannel(Channel):
        """Discrete fourier transform x = FFT z.

        Parameters
        ----------
        - real: bool
          If z supposed to be real

        Notes
        -----
        The fft and ifft are scaled by sqrt(N) so that both are unitary.

        For message passing it is more convenient to represent a complex array x
        as a real array X where X[0] = x.real and X[1] = x.imag

        In particular:
        - output of sample(): X array of shape (2, x.shape)
        - message bx, posterior rx: real arrays of shape (2, x.shape)

        And if real=False (z complex):
          - input  of sample(): Z array of shape (2, z.shape)
          - message bz, posterior rz: real arrays of shape (2, z.shape)
        """

    def __init__(self, real=True):
        self.real = real
        self.repr_init()

    def sample(self, Z):
        "When real=False we assume Z[0] = Z.real and Z[1] = Z.imag"
        if not self.real:
            Z = array2complex(Z)
        X = fftn(Z, norm="ortho")
        X = complex2array(X)
        return X

    def math(self):
        return r"$\mathcal{F}$"

    def second_moment(self, tau):
        return tau

    def compute_forward_message(self, az, bz, ax, bx):
        # x = U z
        ax_new = az
        if not self.real:
            bz = array2complex(bz)
        bx_new = fftn(bz, norm="ortho")
        bx_new = complex2array(bx_new)
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        # z = U.H x
        az_new = ax
        bx = array2complex(bx)
        bz_new = ifftn(bx, norm="ortho")
        if self.real:
            bz_new = np.real(bz_new)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        az_new = ax
        return az_new

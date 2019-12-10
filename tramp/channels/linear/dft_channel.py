import numpy as np
from numpy.fft import fftn, ifftn
from ..base_channel import Channel
from tramp.utils.misc import complex2array, array2complex


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

    def second_moment(self, tau_z):
        return tau_z

    def compute_forward_message(self, az, bz, ax, bx):
        # x = FFT z
        ax_new = az
        if not self.real:
            bz = array2complex(bz)
        bx_new = fftn(bz, norm="ortho")
        bx_new = complex2array(bx_new)
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        # z = IFFT x
        az_new = ax
        bx = array2complex(bx)
        bz_new = ifftn(bx, norm="ortho")
        if self.real:
            bz_new = np.real(bz_new)
        else:
            bz_new = complex2array(bz_new)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        ax_new = az
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        az_new = ax
        return az_new

    def compute_log_partition(self, az, bz, ax, bx):
        _, bz_new = self.compute_backward_mean(az, bz, ax, bx)
        b = bz + bz_new
        a = az + ax
        coef = 0.5 if self.real else 1
        logZ = 0.5 * np.sum(b**2 / a) + coef * self.N * np.log(2 * np.pi / a)
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        a = ax + az
        I = 0.5*np.log(a*tau_z)
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

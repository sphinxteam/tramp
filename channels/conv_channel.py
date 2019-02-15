import numpy as np
from numpy.fft import fftn, ifftn
from ..base import Channel
from ..utils.conv_filters import (
    gaussian_filter, differential_filter, laplacian_filter
)
from ..utils.misc import complex2array, array2complex
import logging


class ConvChannel(Channel):
    """Conv (complex or real) channel x = w * z.

    Parameters
    ----------
    - filter: real or complex array
        Filter weights. The conv weights w are given by w[u] = f*[-u].
        The conv and filter weights ffts are conjugate.
    - real: bool
        if True assume x, w, z real
        if False assume x, w, z complex

    Notes
    -----
    For message passing it is more convenient to represent a complex array x
    as a real array X where X[0] = x.real and X[1] = x.imag

    In particular when real=False (x, w, z complex):
    - input  of sample(): Z real array of shape (2, z.shape)
    - output of sample(): X real array of shape (2, x.shape)
    - message bz, posterior rz: real arrays of shape (2, z.shape)
    - message bx, posterior rx: real arrays of shape (2, x.shape)
    """

    def __init__(self, filter, real=True):
        self.shape = filter.shape
        self.real = real
        self.repr_init()
        self.filter = filter
        # conv weights and filter ffts are conjugate
        self.w_fft_bar = fftn(filter)
        self.w_fft = np.conjugate(self.w_fft_bar)
        self.spectrum = np.absolute(self.w_fft)**2

    def convolve(self, z):
        "We assume x,z,w complex for complex fft or x,w,z real for real fft"
        z_fft = fftn(z)
        x_fft = self.w_fft * z_fft
        x = ifftn(x_fft)
        if self.real:
            x = np.real(x)
        return x

    def sample(self, Z):
        "When real=False we assume Z[0] = Z.real and Z[1] = Z.imag"
        if not self.real:
            Z = array2complex(Z)
        X = self.convolve(Z)
        if not self.real:
            X = complex2array(X)
        return X

    def math(self):
        return r"$\ast$"

    def second_moment(self, tau):
        return tau * self.spectrum.mean()

    def compute_n_eff(self, az, ax):
        "Effective number of parameters = overlap in z"
        if ax == 0:
            logging.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logging.info(f"az/ax=0 in {self} compute_n_eff")
            return 1.
        n_eff = np.mean(self.spectrum / (az / ax + self.spectrum))
        return n_eff

    def compute_backward_mean(self, az, bz, ax, bx, return_fft=False):
        # estimate z from x = Wz
        if not self.real:
            bz = array2complex(bz)
            bx = array2complex(bx)
        bx_fft = fftn(bx)
        bz_fft = fftn(bz)
        resolvent = 1 / (az + ax * self.spectrum)
        rz_fft = resolvent * (bz_fft + self.w_fft_bar * bx_fft)
        if return_fft:
            return rz_fft
        rz = ifftn(rz_fft)
        if self.real:
            rz = np.real(rz)
        else:
            rz = complex2array(rz)
        return rz

    def compute_forward_mean(self, az, bz, ax, bx):
        # estimate x from x = Wz we have rx = W rz
        rz_fft = self.compute_backward_mean(az, bz, ax, bx, return_fft=True)
        rx_fft = self.w_fft * rz_fft
        rx = ifftn(rx_fft)
        if self.real:
            rx = np.real(rx)
        else:
            rx = complex2array(rx)
        return rx

    def compute_backward_variance(self, az, ax):
        if az == 0:
            logging.info(f"az=0 in {self} compute_backward_variance")
            i_mean = np.mean(1 / self.spectrum)
            return i_mean / ax
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_variance(self, az, ax):
        if ax == 0:
            logging.info(f"ax=0 in {self} compute_forward_variance")
            s_mean = np.mean(self.spectrum)
            return s_mean / az
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / ax
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


class DifferentialChannel(ConvChannel):
    def __init__(self, D1, D2, shape, real=True):
        self.D1 = D1
        self.D2 = D2
        self.repr_init()
        f = differential_filter(shape=shape, D1=D1, D2=D2)
        super().__init__(filter=f, real=real)

    def math(self):
        return r"$\partial$"


class LaplacianChannel(ConvChannel):
    def __init__(self, shape, real=True):
        self.repr_init()
        f = laplacian_filter(shape)
        super().__init__(filter=f, real=real)

    def math(self):
        return r"$\Delta$"


class Blur1DChannel(ConvChannel):
    def __init__(self, sigma, N, real=True):
        self.sigma = sigma
        self.repr_init()
        f = gaussian_filter(sigma=sigma, N=N)
        super().__init__(filter=f, real=real)


class Blur2DChannel(ConvChannel):
    def __init__(self, sigma, shape, real=True):
        if len(sigma)!=2:
            raise ValueError("sigma must be a length 2 array")
        if len(shape)!=2:
            raise ValueError("shape must be a length 2 tuple")
        self.sigma = sigma
        self.repr_init()
        f0 = gaussian_filter(sigma=sigma[0], N=shape[0])
        f1 = gaussian_filter(sigma=sigma[1], N=shape[1])
        f = np.outer(f0, f1)
        super().__init__(filter=f, real=real)

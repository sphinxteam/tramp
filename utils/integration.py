import numpy as np
from scipy.integrate import quad, dblquad
from numpy.linalg import cholesky
from .misc import norm_pdf
import logging

def is_pos_def(x):
    return np.all(np.linalg.eigvalsh(x) > 0)

def gaussian_measure(m, s, f):
    """Computes one-dimensional gaussian integral.

    Parameters
    ----------
    - m, s : mean and std of gaussian measure
    - f : function (R -> R) to integrate

    Returns
    -------
    - integral of N(x | m, s) f(x)
    """
    def integrand(x):
        return norm_pdf(x) * f(m + s * x)
    integral = quad(integrand, -10, 10)[0]
    return integral

def gaussian_measure_2d(m1, s1, m2, s2, f):
    """Computes two-dimensional gaussian integral.

    Parameters
    ----------
    - m1, s1 : mean and std of gaussian measure 1st dimension
    - m2, s2 : mean and std of gaussian measure 2nd dimension
    - f : function (R^2 -> R) to integrate

    Returns
    -------
    - integral of N(x1 | m1, s1) N(x2 | m2, s2) f(x1, x2)
    """
    def integrand(x2, x1):
        return norm_pdf(x1) * norm_pdf(x2) * f(m1 + s1 * x1, m2 + s2 * x2)
    integral = dblquad(integrand, -10, 10, -10, 10)[0]
    return integral


def gaussian_measure_2d_full(cov, mean, f):
    """Computes 2-dimensional gaussian integral (full covariance).

    Parameters
    ----------
    - mean : float or array of size 2
        mean vector
    - cov : array of shape (2, 2)
        full covariance matrix
    - f : function (R^2 -> R) to integrate

    Returns
    -------
    - integral of N(x1, x2 | m, cov) f(x1, x2)
    """
    if not is_pos_def(cov):
        logging.warn(f"cov={cov} not pos def")
    L = cholesky(cov)
    def integrand(x2, x1):
        y1, y2 = L @ [x1, x2] + mean
        return norm_pdf(x1) * norm_pdf(x2) * f(y1, y2)
    integral = dblquad(integrand, -10, 10, -10, 10)[0]
    return integral

import numpy as np
from scipy.special import erf, erfcx
import warnings


def gather(df, melted_columns, value_name="value", var_name="variable"):
    """Gather melted_columns."""
    id_vars = [column for column in df.columns if column not in melted_columns]
    melted = df.melt(id_vars=id_vars, value_name=value_name, var_name=var_name)
    return melted


def complex2array(z):
    "Transforms complex z into real array Z where Z[0] = z.real Z[1] = z.imag"
    Z_shape = (2,) + z.shape
    Z = np.zeros(Z_shape)
    Z[0] = z.real
    Z[1] = z.imag
    return Z


def array2complex(Z):
    "Transforms real array Z into complex z where z.real = Z[0] z.imag = Z[1]"
    if Z.shape[0] != 2:
        raise ValueError("First axis of Z must be of length 2")
    z = Z[0] + 1j * Z[1]
    return z


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, slope):
    return np.where(x < 0, slope * x, x)


def hard_tanh(x):
    return np.clip(x, -1, 1)


def hard_sigm(x):
    return np.clip(0.5 + x / 6, 0, 1)


def symm_door(x, width):
    return np.where(np.abs(x) < width, -1, 1)


def norm_cdf(x):
    "Computes Phi(x)"
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def norm_pdf(x):
    "Computes N(x)"
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def log_norm_cdf_prime(x):
    "Computes (log Phi)'(x) = N(x)/Phi(x)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.sqrt(2 * np.pi) * 0.5 * erfcx(-x / np.sqrt(2))
    return 1. / d


# TODO get rid of phi_0, phi_1, phi_2
def phi_0(x):
    "Computes phi(x) = x**2 / 2 + log Phi"
    return np.log(0.5 * erfcx(-x / np.sqrt(2)))


def phi_1(x):
    "Computes phi'(x) = x + N/Phi"
    y = log_norm_cdf_prime(x)
    return x + y


def phi_2(x):
    "Computes phi''(x) = 1 - N/Phi * (x + N/Phi)"
    y = log_norm_cdf_prime(x)
    return 1 - y * (x + y)

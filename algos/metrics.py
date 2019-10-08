import numpy as np
from ..utils.misc import complex2array, array2complex


def mean_squared_error(x_true, x_pred):
    return np.mean((x_true - x_pred)**2)


def sign_symmetric_mse(x_true, x_pred):
    "Mean squared error up to a global sign"
    mse_pos = np.mean((x_true - x_pred) ** 2)
    mse_neg = np.mean((x_true + x_pred) ** 2)
    mse = min(mse_pos, mse_neg)
    return mse


def phase_symmetric_mse(x_true, x_pred):
    "Mean squared error up to a global phase"
    mses = []
    for phi in np.linspace(0, 2*np.pi, 100):
        x_phase = complex2array(np.exp(phi*1j)*array2complex(x_pred))
        mses.append(mean_squared_error(x_true, x_phase))
    mse = min(mses)
    return mse


def overlap(x_true, x_pred):
    return np.mean(x_true * x_pred)


METRICS = {
    "sign_mse": sign_symmetric_mse,
    "phase_mse": phase_symmetric_mse,
    "mse": mean_squared_error,
    "overlap": overlap
}

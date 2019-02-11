import numpy as np


def mean_squared_error(x_true, x_pred):
    return np.mean((x_true - x_pred)**2)


def overlap(x_true, x_pred):
    return np.mean(x_true * x_pred)


METRICS = {
    "mse": mean_squared_error,
    "overlap": overlap
}

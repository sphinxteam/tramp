import numpy as np


def A(b):
    return -np.log(-b)


def r(b):
    return -1 / b


def v(b):
    return 1 / b**2

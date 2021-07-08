import numpy as np
from ..utils.truncated_normal import (
    truncated_normal_mean, truncated_normal_var, truncated_normal_logZ,
    truncated_normal_proba
)


def A(a, b):
    return truncated_normal_logZ(b/a, 1/a, 0, np.inf)


def r(a, b):
    return truncated_normal_mean(b/a, 1/a, 0, np.inf)


def v(a, b):
    return truncated_normal_var(b/a, 1/a, 0, np.inf)


def tau(a, b):
    return r(a, b)**2 + v(a, b)


def p(a, b):
    "Probabilty that x ~ N(r, v) with r=b/a and v=1/a falls within R_+"
    return truncated_normal_proba(b/a, 1/a, 0, np.inf)

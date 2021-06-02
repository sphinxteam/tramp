from ..utils.truncated_normal import (
    truncated_normal_mean, truncated_normal_var, truncated_normal_logZ
)


def A(a, b, xmin, xmax):
    return truncated_normal_logZ(b / a, a, xmin, xmax)


def r(a, b, xmin, xmax):
    return truncated_normal_mean(b / a, a, xmin, xmax)


def v(a, b, xmin, xmax):
    return truncated_normal_var(b / a, a, xmin, xmax)

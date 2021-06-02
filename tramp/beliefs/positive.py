from ..utils.truncated_normal import (
    truncated_normal_mean, truncated_normal_var, truncated_normal_logZ
)


def A(a, b):
    return truncated_normal_logZ(b / a, a, 0, np.inf)


def r(a, b):
    return truncated_normal_mean(b / a, a, 0, np.inf)


def v(a, b):
    return truncated_normal_var(b / a, a, 0, np.inf)

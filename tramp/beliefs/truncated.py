from ..utils.truncated_normal import (
    truncated_normal_mean, truncated_normal_var, truncated_normal_logZ,
    truncated_normal_proba
)


def A(a, b, xmin, xmax):
    return truncated_normal_logZ(b/a, 1/a, xmin, xmax)


def r(a, b, xmin, xmax):
    return truncated_normal_mean(b/a, 1/a, xmin, xmax)


def v(a, b, xmin, xmax):
    return truncated_normal_var(b/a, 1/a, xmin, xmax)


def tau(a, b, xmin, xmax):
    return r(a, b, xmin, xmax)**2 + v(a, b, xmin, xmax)


def p(a, b, xmin, xmax):
    "Probabilty that x ~ N(r, v) with r=b/a and v=1/a falls within [xmin, xmax]"
    return truncated_normal_proba(b/a, 1/a, xmin, xmax)

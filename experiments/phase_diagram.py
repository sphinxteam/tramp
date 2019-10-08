from ..algos import StateEvolution, CustomInit, ConstantInit
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def binary_search(f, xmin, xmax, xtol=1e-6, max_iter=100):
    "Binary search on boolean f, assuming f(xmin)=0 and f(xmax)=1"
    ymin, ymax = f(xmin), f(xmax)
    if not (ymin == 0 and ymax == 1):
        raise ValueError(f"Bad bounds: ymin={ymin} and ymax={ymax}")
    for iter in range(max_iter):
        xmid = (xmin + xmax)/2
        ymid = f(xmid)
        xerr = xmax - xmin
        logger.info(f"binary search {iter}/{max_iter} xerr={xerr}")
        if (xerr < xtol):
            break
        if ymid == 0:
            xmin, ymin = xmid, ymid
        else:
            xmax, ymax = xmid, ymid
    assert ymin == 0 and ymax == 1
    if (iter == max_iter-1):
        logger.warn(f"binary search did not converge xerr={xerr}")
    return dict(
        xmid=xmid, xmin=xmin, xmax=xmax, iter=iter, xerr=xerr
    )


def get_se(id, a0, alpha, model_builder, **model_kwargs):
    model = model_builder(alpha=alpha, **model_kwargs)
    if a0 == 0:
        initializer = ConstantInit(a=0)
    else:
        a_init = [(id, "bwd", a0)]
        initializer = CustomInit(a_init=a_init)
    se = StateEvolution(model)
    se.iterate(
        max_iter=200, initializer=initializer, check_decreasing=False
    )
    v = se.get_variable_data(id=id)["v"]
    return dict(v=v, n_iter=se.n_iter)


def get_mse(id, a0, alpha, model_builder, **model_kwargs):
    model = model_builder(alpha=alpha, **model_kwargs)
    if a0 == 0:
        initializer = ConstantInit(a=0)
    else:
        a_init = [(id, "bwd", a0)]
        initializer = CustomInit(a_init=a_init)
    se = StateEvolution(model)
    se.iterate(
        max_iter=200, initializer=initializer, check_decreasing=False
    )
    v = se.get_variable_data(id=id)["v"]
    return v


def find_critical_alpha(id, a0, mse_criterion, alpha_min, alpha_max,
                        model_builder,
                        xtol=1e-6, max_iter=100, return_search=False,
                        **model_kwargs):
    def f(alpha):
        v = get_mse(id, a0, alpha, model_builder, **model_kwargs)
        return mse_criterion(v)

    search = binary_search(f, alpha_min, alpha_max, xtol, max_iter)
    if return_search:
        return search
    alpha_c = search["xmid"]
    return alpha_c

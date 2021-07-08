from ..algos import StateEvolution, CustomInit, ConstantInit
import numpy as np
import logging
logger = logging.getLogger(__name__)


def binary_search(f, xmin, xmax, xtol):
    "Binary search on boolean f, assuming f(xmin)=0 and f(xmax)=1"
    ymin, ymax = f(xmin), f(xmax)
    if not (ymin == 0 and ymax == 1):
        raise ValueError(f"Bad bounds: ymin={ymin} and ymax={ymax}")
    max_iter = int(np.log2((xmax - xmin) / xtol)) + 2
    for n_iter in range(1, max_iter + 1):
        xmid = (xmin + xmax)/2
        ymid = f(xmid)
        xerr = xmax - xmin
        logger.info(f"binary search {n_iter}/{max_iter} xerr={xerr}")
        if (xerr < xtol):
            break
        if ymid == 0:
            xmin, ymin = xmid, ymid
        else:
            xmax, ymax = xmid, ymid
    assert ymin == 0 and ymax == 1
    assert (xerr < xtol)
    return dict(
        xmid=xmid, xmin=xmin, xmax=xmax, xerr=xerr, n_iter=n_iter
    )


def find_state_evolution_mse(id, a0, alpha, model_builder, **model_kwargs):
    """Find the variable mse according to the state evolution of the model.

    Parameters
    ----------
    id : str
        id of the variables to infer (signal)
    a0 : float
        initial value of the a message id -> prior
    alpha : float
        measurement density
    model_builder : function or class
    **model_kwargs : dict
        model_builder(**model_kwargs) must return a Model instance.

    Returns
    -------
    v : float
        The variable mse according to state evolution
    """
    model = model_builder(alpha=alpha, **model_kwargs)
    a_init = [(id, "bwd", a0)]
    initializer = CustomInit(a_init=a_init)
    se = StateEvolution(model)
    se.iterate(max_iter=200, initializer=initializer)
    v = se.get_variable_data(id=id)["v"]
    return v


def find_critical_alpha(id, a0, mse_criterion, alpha_min, alpha_max,
                        model_builder, alpha_tol=1e-6, vtol=1e-3,
                        **model_kwargs):
    """Find critical value of the measurment density alpha.

    It performs a binary search on alpha to find the minimal value of alpha for
    which the mse criterion is satisfied.

    Parameters
    ----------
    id : str
        id of the variable to infer (signal)
    a0 : float
        Initial value of the a message id -> prior
    mse_criterion : {"random", "perfect"} or function
        Criterion on the mse:

        - "random" : search the maximal value of alpha for which v = tau_x (no better than random guess)
        - "perfect" : search the minimal value of alpha for which v = 0 (perfect reconstruction)
        - function : mse_criterion(v) must return False when alpha < alpha_c and True when alpha > alpha_c

    alpha_min : float
        Minimal value for the alpha search
    alpha_max : float
        Maximal value for the alpha search
    alpha_tol : float,
        Tolerance on alpha, default 1e-6
    vtol : float
        Tolerance on the variance v used in the "perfect" or "random" mse
        criteria
    """
    if mse_criterion == "perfect":
        def mse_criterion(v):
            return abs(v) < vtol
    elif mse_criterion == "random":
        # assuming that tau_x does not depend on alpha, we choose a fixed value
        model = model_builder(alpha=0.5, **model_kwargs)
        model.init_second_moments()
        tau_x = model.get_second_moments()[id]

        def mse_criterion(v):
            return abs(v - tau_x) > vtol

    def f(alpha):
        v = find_state_evolution_mse(id, a0, alpha, model_builder, **model_kwargs)
        return mse_criterion(v)

    search = binary_search(f, alpha_min, alpha_max, alpha_tol)
    alpha_c = search["xmid"]
    return alpha_c

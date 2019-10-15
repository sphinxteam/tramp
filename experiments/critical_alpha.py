from ..algos import StateEvolution, CustomInit, ConstantInit
import logging
logger = logging.getLogger(__name__)


def binary_search(f, xmin, xmax, xtol=1e-6, max_iter=100):
    "Binary search on boolean f, assuming f(xmin)=0 and f(xmax)=1"
    ymin, ymax = f(xmin), f(xmax)
    if not (ymin == 0 and ymax == 1):
        raise ValueError(f"Bad bounds: ymin={ymin} and ymax={ymax}")
    for n_iter in range(max_iter):
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
    if (n_iter == max_iter-1):
        logger.warn(f"binary search did not converge xerr={xerr}")
    return dict(
        xmid=xmid, xmin=xmin, xmax=xmax, xerr=xerr, n_iter=n_iter
    )


def find_state_evolution_mse(id, a0, alpha, model_builder, **model_kwargs):
    """
    Find the variable mse according to the state evolution of the model.

    Parameters
    ----------
    - id : id of the variables to infer (signal)
    - a0 : initial value of the a message id -> prior
    - alpha : measurment density
    - model_builder : function or class
    - model_kwargs : model arguments
        model_builder(**model_kwargs) must return a Model instance.
    Returns
    -------
    - result : dict
    """
    model = model_builder(alpha=alpha, **model_kwargs)
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
                        alpha_tol=1e-6, max_iter=100, return_search=False,
                        vtol=1e-3,
                        **model_kwargs):
    """
    Find critical value of the measurment density alpha. It performs a binary
    search on alpha to find the minimal value of alpha for which the mse
    criterion is satisfied.

    Parameters
    ----------
    - id : id of the variables to infer (signal)
    - a0 : initial value of the a message id -> prior
    - mse_criterion : str or function
        - "random" : search the maximal value of alpha for which v = tau_x
        (no better than random guess)
        - "perfect" : search the minimal value of alpha for which v = 0,
        (perfect reconstruction)
        - function : choosen so that mse_criterion(v) returns False when
        alpha < alpha_c and True when alpha > alpha_c
    - alpha_min, alpha_max : min and max values for the alpha search.
    - alpha_tol : tolerance on alpha, default 1e-6
    - max_iter : maximal number of iterations for the binary search
    - return_search : boolean, whether to return search result or only alpha
    - vtol : tolerance on the variance v used in the "perfect" or "random" mse
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

    search = binary_search(f, alpha_min, alpha_max, alpha_tol, max_iter)
    if return_search:
        return search
    alpha_c = search["xmid"]
    return alpha_c

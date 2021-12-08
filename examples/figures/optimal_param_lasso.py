import logging
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Lasso
from tramp.ensembles import GaussianEnsemble
from tramp.priors import GaussBernoulliPrior
from tramp.channels import GaussianChannel, LinearChannel
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.algos.metrics import mean_squared_error
from tramp.experiments import BayesOptimalScenario, save_experiments


def mse_lasso(alpha, param_scaled, seed):
    # create scenario
    N, rho, noise_var = 1000, 0.05, 1e-2
    M = int(alpha*N)
    A = GaussianEnsemble(M=M, N=N).generate()
    model = (
        GaussBernoulliPrior(size=N, rho=rho) @ V("x") @
        LinearChannel(A) @ V("z") @ GaussianChannel(var=noise_var) @ O("y")
    ).to_model()
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    scenario.setup(seed)
    y = scenario.observations["y"]
    # run lasso
    param_scikit = noise_var * param_scaled / (M * rho)
    lasso = Lasso(alpha=param_scikit)
    lasso.fit(A, y)
    x_pred = lasso.coef_
    mse = mean_squared_error(x_pred, scenario.x_true["x"])
    return mse


def find_optimal_param(alpha):
    def average_mse(param_scaled):
        mses = [mse_lasso(alpha, param_scaled, seed) for seed in np.arange(100)]
        return np.mean(mses)
    res = minimize_scalar(
        average_mse, bounds=(0.1, 0.9), method='bounded',
        options={"xatol": 0.001, "disp": 1}
    )
    return dict(param_scaled=res.x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv_file = __file__.replace(".py", ".csv")
    save_experiments(
        find_optimal_param, csv_file, alpha=np.linspace(0.02, 1, 50)
    )

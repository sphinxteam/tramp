import logging
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
from time import time
from sklearn.linear_model import Lasso, LassoCV
from tramp.ensembles import GaussianEnsemble
from tramp.priors import GaussBernoulliPrior
from tramp.channels import GaussianChannel, LinearChannel
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.algos.metrics import mean_squared_error
from tramp.experiments import BayesOptimalScenario, save_experiments


def run_benchmark(alpha, algo, seed):
    # create scenario
    N, rho, noise_var = 1000, 0.05, 1e-2
    M = int(alpha*N)
    A = GaussianEnsemble(M=M, N=N).generate()
    t0 = time()
    model = (
        GaussBernoulliPrior(size=N, rho=rho) @ V("x") @
        LinearChannel(A) @ V("z") @ GaussianChannel(var=noise_var) @ O("y")
    ).to_model()
    t1 = time()
    record = {"svd_time": t1 - t0}  # svd precomputation time
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    scenario.setup(seed)
    y = scenario.observations["y"]
    # run algo
    t0 = time()
    if algo == "SE":
        x_data = scenario.run_se(max_iter=1000, damping=0.1)
        record["mse"] = x_data["x"]["v"]
        record["n_iter"] = x_data["n_iter"]
    if algo == "EP":
        x_data = scenario.run_ep(max_iter=1000, damping=0.1)
        x_pred = x_data["x"]["r"]
        record["n_iter"] = x_data["n_iter"]
    if algo == "LassoCV":
        lasso = LassoCV(cv=5)
        lasso.fit(A, y)
        x_pred = lasso.coef_
        record["param_scikit"] = lasso.alpha_
        record["n_iter"] = lasso.n_iter_
    if algo == "Lasso":
        optim = pd.read_csv("optimal_param_lasso.csv")
        param_scaled = np.interp(alpha, optim["alpha"], optim["param_scaled"])
        param_scikit = noise_var * param_scaled / (M * rho)
        lasso = Lasso(alpha=param_scikit)
        lasso.fit(A, y)
        x_pred = lasso.coef_
        record["param_scikit"] = param_scikit
        record["n_iter"] = lasso.n_iter_
    if algo == "pymc3":
        with pm.Model():
            ber = pm.Bernoulli("ber", p=rho, shape=N)
            nor = pm.Normal("nor", mu=0, sd=1, shape=N)
            x = pm.Deterministic("x", ber * nor)
            likelihood = pm.Normal(
                "y", mu=pm.math.dot(A, x), sigma=np.sqrt(noise_var), observed=y
            )
            trace = pm.sample(draws=1000, chains=1, return_inferencedata=False)
        x_pred = trace.get_values('x').mean(axis=0)
    t1 = time()
    record["time"] = t1-t0
    if algo != "SE":
        record["mse"] = mean_squared_error(x_pred, scenario.x_true["x"])
    return record


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # cmd line options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", choices=["SE", "EP", "Lasso", "LassoCV", "pymc3"],
        help="algo to benchmark"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="run seed = SEED")
    group.add_argument("--N", type=int, help="run seed in range(N)")
    args = parser.parse_args()
    # run benchmark algo and seed = int or range
    seed = args.seed if args.N is None else np.arange(args.N)
    seed_str = f"seed{seed:03}" if args.N is None else f"N{args.N}"
    csv_file = f"benchmark-{args.algo}-{seed_str}.csv"
    logging.info(f"Running seed={args.seed} N={args.N} algo={args.algo}")
    save_experiments(
        run_benchmark, csv_file,
        algo=args.algo, alpha=np.linspace(0.02, 1, 50), seed=seed
    )

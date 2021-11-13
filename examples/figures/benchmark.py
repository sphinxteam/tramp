import logging
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
from time import time
from sklearn.linear_model import LassoCV
from tramp.ensembles import GaussianEnsemble
from tramp.priors import GaussBernoulliPrior
from tramp.channels import GaussianChannel, LinearChannel
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.algos.metrics import mean_squared_error
from tramp.experiments import BayesOptimalScenario, save_experiments
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=24)


def plot_benchmark():
    algos = ["SE", "EP"]
    # read data
    df = pd.concat(
        [pd.read_csv(f"benchmark-{algo}.csv") for algo in algos], ignore_index=True
    ).groupby(["alpha", "algo"]).mean().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    label = {
        "SE":"Bayes opt.", "EP":"Tree-AMP", "Lasso":"Lasso", "pymc3":"PyMC3"
    }
    marker = {"SE":"k-", "EP":"x--", "Lasso":"^--", "pymc3":"o"}
    for algo in algos:
        df_algo = df[df.algo == algo]
        rho = 0.05
        axs[0].plot(df_algo["alpha"], df_algo["mse"]/rho, marker[algo], label=label[algo])
        if algo != "SE":
            axs[1].plot(df_algo["alpha"], df_algo["time"], marker[algo], label=label[algo])
    axs[0].legend()
    axs[0].set(xlabel=r'$\alpha$', ylabel=r'MSE / $\rho$', xlim=[0, 1], ylim=[0, 1.1])
    axs[1].legend()
    axs[1].set(xlabel=r'$\alpha$', ylabel=r'Time $(s)$', xlim=[0, 1], yscale="log")
    fig.tight_layout()
    logging.info("Saving benchmark.pdf")
    plt.savefig(
        "benchmark.pdf", format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.1
    )


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
    if algo == "Lasso":
        lasso = LassoCV(cv=5)
        lasso.fit(A, y)
        x_pred = lasso.coef_
        record["alpha_"] = lasso.alpha_
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
    # cmd line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="plot and save figure")
    parser.add_argument("--algo", choices=["SE", "EP", "Lasso", "pymc3"], help="algo to benchmark")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="run seed = SEED")
    group.add_argument("--N", type=int, help="run seed in range(N)")
    args = parser.parse_args()
    # plot
    if args.plot:
        plot_benchmark()
        exit()
    # seed = int or range
    seed = args.seed if args.N is None else np.arange(args.N)
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running seed={seed} algo={args.algo}")
    save_experiments(
        run_benchmark, f"benchmark-{args.algo}.csv",
        algo=args.algo, alpha=np.linspace(0.02, 1, 50), seed=seed
    )

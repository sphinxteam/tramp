import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=24)


def compute_average():
    csv_files = [
        file for file in os.listdir(".")
        if file.startswith(f"benchmark-") and file.endswith(".csv")
    ]
    df = pd.concat(
        [pd.read_csv(csv_file) for csv_file in csv_files], ignore_index=True
    )
    funs = {col: "mean" for col in df.columns if col not in ["alpha", "algo"]}
    funs["seed"] = "nunique"
    df_avg = df.groupby(
        ["algo", "alpha"], as_index=False
    ).agg(funs).rename(columns={"seed":"n_seed"})
    logging.info("Saving benchmark.csv")
    df_avg.to_csv("benchmark.csv", index=False)


def plot_benchmark():
    df = pd.read_csv("benchmark.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    label = {
        "SE": "Bayes opt.", "EP": "Tree-AMP", "Lasso": "Lasso", "pymc3": "PyMC3"
    }
    marker = {"SE": "k-", "EP": "x--", "Lasso": "^--", "pymc3": "o"}
    for algo in df.algo.unique():
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "benchmark.csv" not in os.listdir():
        compute_average()
    plot_benchmark()

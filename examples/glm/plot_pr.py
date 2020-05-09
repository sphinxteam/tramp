"""
Phase retrieval
===============

"""

# %%
# Setup
from tramp.algos import EarlyStoppingEP
from tramp.experiments import BayesOptimalScenario, qplot, plot_compare_complex
from tramp.models import glm_generative
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %%
# Model
np.random.seed(42)

model = glm_generative(
    N=1000, alpha=2, ensemble_type="complex_gaussian",
    prior_type="gauss_bernouilli", output_type="modulus",
    prior_mean=0.01, prior_rho=0.5
)
scenario = BayesOptimalScenario(model, x_ids=["x"])
scenario.setup()
scenario.student.plot()

for factor in scenario.student.factors:
    print(factor.id, factor)


# %%
# EP dyanmics
# Damping is needed
# really bad without damping
ep_evo = scenario.ep_convergence(
    metrics=["mse", "phase_mse"], max_iter=20
)
qplot(
    ep_evo, x="iter", y=["phase_mse", "v"],
    y_markers=["x", "-"], y_legend=True
)
ep_evo = scenario.ep_convergence(
    metrics=["mse", "phase_mse"], max_iter=70, damping=0.3
)
qplot(
    ep_evo, x="iter", y=["mse", "phase_mse", "v"],
    y_markers=[".", "x", "-"], y_legend=True
)

plot_compare_complex(scenario.x_true["x"], scenario.x_pred["x"])


# %%
# Compare EP vs SE
rename = {
    "alpha": r"$\alpha$", "prior_mean": r"$\mu$", "prior_rho": r"$\rho$",
    "n_iter": "iterations", "source=": "", "phase_mse": "p-mse",
    "a0=0.1": "uninformed", "a0=1000.0": "informed"
}
ep_vs_se = pd.read_csv("data/phase_retrieval_ep_vs_se.csv")
qplot(
    ep_vs_se.query("source!='mse'"), x="alpha", y="v", marker="source", column="prior_mean",
    rename=rename, usetex=True
)
qplot(
    ep_vs_se.query("source=='SE'"),
    x="alpha", y="v", color="prior_mean",
    rename=rename, usetex=True
)
qplot(
    ep_vs_se.query("source=='SE'"),
    x="alpha", y="n_iter", color="prior_mean",
    rename=rename, usetex=True
)

# %%
# MSE curves
mse_curves = pd.read_csv("data/phase_retrieval_mse_curves.csv")
qplot(
    mse_curves, x="alpha", y="v", linestyle="a0", column="prior_rho",
    rename=rename, usetex=True, font_size=16
)

"""
Relu
====

"""

# %%
# Setup
from tramp.experiments import run_experiments, qplot, plot_compare
from tramp.models import glm_generative
from tramp.experiments import BayesOptimalScenario
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)


# %%
# Model

alpha = 2.
N = 1000
teacher = glm_generative(
    N=N, alpha=alpha, ensemble_type="gaussian", prior_type="gauss_bernoulli",
    output_type="relu", prior_rho=0.5
)
for factor in teacher.factors:
    print(factor)
scenario = BayesOptimalScenario(teacher, x_ids=["x", "z"])
scenario.setup(seed=42)
scenario.student.plot()


# %%
# EP dyanmics
ep_evo = scenario.ep_convergence(metrics=["mse"], max_iter=10)
qplot(
    ep_evo, x="iter", y=["mse", "v"],
    y_markers=[".", "-"], column="id", y_legend=True
)
plot_compare(scenario.x_true["x"], scenario.x_pred["x"])

# %%
# MSE curve
# See data/relu_mse_curves.py for the code
rename = {
    "alpha": r"$\alpha$", "v": "MSE", "n_iter": "iter", "a0": r"$a_0$",
    "prior_rho": r"$\rho$", "x_id=": "", "n_iter": "iterations"
}
mse_curves = pd.read_csv("data/relu_mse_curves.csv")

qplot(
    mse_curves.query("x_id =='x'"), x="alpha", y="v", color="prior_rho",
    rename=rename, usetex=True, font_size=14
)
qplot(
    mse_curves.query("x_id =='x'"), x="alpha", y="n_iter", color="prior_rho",
    rename=rename, usetex=True, font_size=14
)

# %%
# Critical lines
# See data/relu_critical_lines.py for the code.
crit = pd.read_csv("data/relu_critical_lines.csv")

qplot(
    crit,
    x="prior_rho", y="alpha", color="a0",
    rename=rename, usetex=True, font_size=14
)
plt.plot([0, 1], [0, 2], color="grey")

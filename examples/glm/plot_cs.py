"""
Compressed Censing
==================

"""

# %%
# Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tramp.experiments import BayesOptimalScenario, qplot, plot_compare
from tramp.models import glm_generative


# %%
# Model
alpha = 0.8
N = 1000
teacher = glm_generative(
    N=N, alpha=alpha, ensemble_type="gaussian", prior_type="gauss_bernoulli",
    output_type="gaussian", output_var=1e-11, prior_rho=0.5
)
for factor in teacher.factors:
    print(factor)
scenario = BayesOptimalScenario(teacher, x_ids=["x"])
scenario.setup()
scenario.student.plot()


# %%
# EP dyanmics
ep_evo = scenario.ep_convergence(metrics=["mse"], max_iter=10)
plot_compare(scenario.x_true["x"], scenario.x_pred["x"])
qplot(
    ep_evo, x="iter", y=["v", "mse"], y_markers=["-", "."],
    y_legend=True
)


# %%
# Compare EP vs SE
rename = {
    "alpha": r"$\alpha$", "prior_rho": r"$\rho$",
    "source=": "", "n_iter": "iterations"
}
ep_vs_se = pd.read_csv("data/compressed_sensing_ep_vs_se.csv")
qplot(
    ep_vs_se, x="alpha", y="v", marker="source", column="prior_rho",
    rename=rename, usetex=True, font_size=16
)

# %%
# MSE curves
qplot(
    ep_vs_se.query("source=='SE'"),
    x="alpha", y="v", color="prior_rho",
    rename=rename, usetex=True, font_size=16
)


# %%
# Number of iterations diverging at the critical value
qplot(
    ep_vs_se.query("source=='SE'"),
    x="alpha", y="n_iter", color="prior_rho",
    rename=rename, usetex=True, font_size=16
)

# %%
# Critcal lines
crit = pd.read_csv("data/cs_critical_lines.csv")
qplot(
    crit,
    x="prior_rho", y="alpha",
    rename=rename, usetex=True, font_size=16
)

# %%
# Universality
# The linear channel update, in the noiseless case, will only depend on the trace of the sensing matrix $F$.  Here we show for random features matrix $F = \tfrac{1}{\sqrt{N}}f(WX)$

univ = pd.read_csv("data/cs_universality.csv")
qplot(
    univ.query("source=='SE'"),
    x="alpha", y="v", color="f", column="prior_rho",
    rename=rename, usetex=True, font_size=16
)

"""
Perceptron
==========

"""

# %%
# Setup
from tramp.algos import EarlyStoppingEP
from tramp.models import glm_generative
from tramp.experiments import BayesOptimalScenario, qplot, plot_compare
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %%
# Model
# You can build the perceptron directly, or use the `glm_generative` model builder.
alpha = 1.7
N = 1000
teacher = glm_generative(
    N=N, alpha=alpha, ensemble_type="gaussian", prior_type="binary",
    output_type="sgn"
)

scenario = BayesOptimalScenario(teacher, x_ids=["x"])
scenario.setup()
scenario.student.plot()

for factor in scenario.student.factors:
    print(factor.id, factor)


# %%
# EP dyanmics
ep_evo = scenario.ep_convergence(
    metrics=["mse"], max_iter=30, callback=EarlyStoppingEP()
)
plot_compare(scenario.x_true["x"], scenario.x_pred["x"])

qplot(
    ep_evo, x="iter", y=["mse", "v"],
    y_markers=[".", "-"], y_legend=True
)


# %%
# Compare EP vs SE
# See `data/perceptron_ep_vs_se.py`
rename = {
    "alpha": r"$\alpha$", "n_iter": "iterations", "p_pos": r"$p_+$", "source=": ""
}
df = pd.read_csv("data/perceptron_ep_vs_se.csv")

qplot(
    df, x="alpha", y="v", marker="source", column="p_pos",
    rename=rename, usetex=True, font_size=16
)


# %%
# Phase transition
qplot(
    df.query("source=='SE'"), x="alpha", y="v", color="p_pos",
    rename=rename, usetex=True, font_size=16
)


# %%
# Number of iterations diverging at the critical value
qplot(
    df.query("source=='SE'"),
    x="alpha", y="n_iter", color="p_pos", column="source",
    rename=rename, usetex=True, font_size=16
)

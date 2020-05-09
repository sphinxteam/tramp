"""
Symmetric door
==============

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
alpha = 1.6
N = 1000
teacher = glm_generative(
    N=N, alpha=alpha, ensemble_type="gaussian", prior_type="binary",
    output_type="door", output_width=1., prior_p_pos=0.51
)

for factor in teacher.factors:
    print(factor)

scenario = BayesOptimalScenario(teacher, x_ids=["x", "z"])
scenario.setup(seed=42)
scenario.student.plot()

# %%
# EP dyanmics
ep_evo = scenario.ep_convergence(
    metrics=["mse", "sign_mse"], damping=0.1, max_iter=40
)
qplot(
    ep_evo, x="iter",
    y=["mse", "sign_mse", "v"],  y_markers=["x", ".", "-"],
    column="id", y_legend=True
)
plot_compare(scenario.x_true["x"], scenario.x_pred["x"])

# %%
# MSE curve
# See data/door_mse_curves.py for the code
rename = {
    "alpha": r"$\alpha$", "output_width": "K",
    "a0": r"$a_0$", "v": "MSE", "n_iter": "iterations",
    "x_id=": "", "p_pos": r"$p_+$", "criterion=": ""
}
mse_curves = pd.read_csv("data/door_mse_curves.csv")
qplot(
    mse_curves, x="alpha", y="v", color="output_width",
    rename=rename, usetex=True, font_size=14
)


# %%
# Critical lines
# See data/door_critical_lines.py for the code.
crit = pd.read_csv("data/door_critical_lines.csv")
# The critical lines seem okay for the  "uninformed" case ($a_0=0.1$) with $p_+=0.51$.
# Informed fixed point a0=1000 failed
qplot(
    crit,
    x="output_width", y="alpha", column="a0", marker="criterion",
    rename=rename, usetex=True, font_size=14
)

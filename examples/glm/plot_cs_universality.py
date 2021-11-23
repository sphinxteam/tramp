"""
Universality (noiseless CS)
===========================

"""
import pandas as pd
from matplotlib import rcParams
rcParams['axes.unicode_minus'] = False
from tramp.experiments import qplot

# %%
# Model
# -----
# We consider for the sensing matrix $F$ a random features matrix
# $F = \tfrac{1}{\sqrt{N}}f(WX)$ where $f$ = abs, relu, sgn or tanh.
# See `data/cs_universality.py` for the corresponding script.
rename = {
    "alpha": r"$\alpha$", "prior_rho": r"$\rho$",
    "source=": "", "n_iter": "iterations"
}
univ = pd.read_csv("data/cs_universality.csv")
qplot(
    univ.query("source=='SE'"),
    x="alpha", y="v", linestyle="f", column="prior_rho",
    rename=rename, usetex=True, font_size=16
)

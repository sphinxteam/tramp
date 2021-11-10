"""
Sparse gradient
===============

"""
from tramp.algos import EarlyStoppingEP
from tramp.experiments import TeacherStudentScenario
from tramp.variables import SISOVariable as V, SILeafVariable as O, MILeafVariable, SIMOVariable
from tramp.channels import GaussianChannel, GradientChannel
from tramp.priors import GaussBernoulliPrior, GaussianPrior
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=24)


# %%
# Define a sparse gradient teacher
class SparseGradTeacher():
    def __init__(self, N, rho, noise_var):
        self.prior_grad = GaussBernoulliPrior(size=(1, N), rho=rho)
        self.channel = GaussianChannel(var=noise_var)

    def sample(self, seed=None):
        if seed:
            np.random.seed(seed)
        x_prime = self.prior_grad.sample()
        x = x_prime.ravel().cumsum()
        x = x - x.mean()
        y = self.channel.sample(x)
        return {"x": x, "z": x_prime, "y": y}


# %%
# Define a sparse gradient student
def build_sparse_grad_student(N, rho, noise_var):
    x_shape = (N,)
    z_shape = (1, N)
    student = (
        GaussianPrior(size=x_shape) @
        SIMOVariable(id="x", n_next=2) @ (
            GaussianChannel(var=noise_var) @ O("y") + (
                GradientChannel(shape=x_shape) +
                GaussBernoulliPrior(size=z_shape, rho=rho)
            ) @ MILeafVariable(id="z", n_prev=2)
        )
    ).to_model()
    return student


# %%
# Plotting function
def plot_sparse_gradient(scenario):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].plot(scenario.observations["y"], 'C2x', label=r'$y$')
    axes[0].set_title(r'$y$')
    axes[1].plot(scenario.x_true["x"], label=r'$x^*$')
    axes[1].plot(scenario.x_pred["x"], label=r'$\hat{x}$')
    axes[1].set_title(r'$x$')
    axes[2].stem(
        scenario.x_true["z"][0], label=r'$z^*$',
        markerfmt="C0o", linefmt="C0-", basefmt="C0", use_line_collection=True
    )
    axes[2].stem(
        scenario.x_pred["z"][0], label=r'$\hat{z}$',
        markerfmt="C1o", linefmt="C1-", basefmt="C1", use_line_collection=True
    )
    axes[2].set_title(r'$z = \nabla x$')
    for axe in axes:
        axe.legend(fancybox=True, shadow=False, loc="lower center", fontsize=20)
    fig.tight_layout()


# %%
# Parameters
size, rho, noise_var, seed = 400, 0.04, 1e-2, 1

# %%
# We create the teacher student scenario
teacher = SparseGradTeacher(size, rho, noise_var)
student = build_sparse_grad_student(size, rho, noise_var)
scenario = TeacherStudentScenario(teacher, student, x_ids=["x", "z"])
scenario.setup(seed=seed)

# %%
# Run EP
_ = scenario.run_ep(
    max_iter=1000, damping=0.1, callback=EarlyStoppingEP(tol=1e-2)
)

# %%
# Plot
plot_sparse_gradient(scenario)

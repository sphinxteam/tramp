import logging
import numpy as np
import pandas as pd
from tramp.models import glm_generative, glm_state_evolution
from tramp.experiments import run_experiments, BayesOptimalScenario
from tramp.algos import CustomInit, StateEvolution
from tramp.algos.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=20)


def plot_discontinuous(ax, x, y, marker, label):
    jump = np.where(y > 1e-3)[0][-1]
    ax.plot(x[:jump+1], y[:jump+1], marker, label=label)
    ax.plot([x[jump], x[jump]], [y[jump], y[jump+1]], marker[:2]+":")
    ax.plot(x[jump+1:], y[jump+1:], marker)


def plot_mse_curves():
    # read data
    csv_file = __file__.replace(".py", ".csv")
    df = pd.read_csv(csv_file)
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    df_ep = df[df.source == "EP"]
    ax.plot(df_ep["alpha"], df_ep["v"], "C1o", label="EP")
    df_se = df[df.source == "SE"]
    plot_discontinuous(
        ax, df_se["alpha"].values, df_se["v"].values, "C0-", label="SE"
    )
    df_se = df[df.source == "BO"]
    plot_discontinuous(
        ax, df_se["alpha"].values, df_se["v"].values, "C2--", label="Bayes opt."
    )
    ax.set(xlabel=r'$\alpha$', ylabel=r'MSE')
    ax.legend(loc='upper right', fancybox=True, shadow=False)
    fig.tight_layout()
    # save figure
    pdf_file = __file__.replace(".py", ".pdf")
    logging.info(f"Saving {pdf_file}")
    plt.savefig(
        pdf_file, format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.1
    )


def run_EP(alpha, rho, seed):
    model = glm_generative(
        N=2000, alpha=alpha, ensemble_type="gaussian",
        prior_type="gauss_bernoulli", output_type="gaussian",
        prior_rho=rho, output_var=1e-10
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    scenario.setup(seed)
    x_data = scenario.run_ep(max_iter=200)
    x_pred = x_data["x"]["r"]
    mse = mean_squared_error(x_pred, scenario.x_true["x"])
    return dict(source="EP", v=mse)


def run_SE(alpha, rho):
    # analytical linear channel (Marcenko Pastur)
    model = glm_state_evolution(
        alpha=alpha, prior_type="gauss_bernoulli", output_type="gaussian",
        prior_rho=rho, output_var=1e-10
    )
    # SE : uninformed initialization
    se = StateEvolution(model)
    se.iterate(max_iter=200)
    x_data = se.get_variable_data(id="x")
    return dict(source="SE", v=x_data["v"])


def run_BO(alpha, rho):
    # analytical linear channel (Marcenko Pastur)
    model = glm_state_evolution(
        alpha=alpha, prior_type="gauss_bernoulli", output_type="gaussian",
        prior_rho=rho, output_var=1e-10
    )
    # BO : informative initialization, scaled to avoid issues at low alpha
    power = 3 * np.exp(alpha)
    initializer = CustomInit(a_init=[("x", "bwd", 10**power)])
    se = StateEvolution(model)
    se.iterate(max_iter=200, initializer=initializer)
    x_data = se.get_variable_data(id="x")
    return dict(source="BO", v=x_data["v"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # avg over 25 instances of EP
    df_EP = run_experiments(
        run_EP, alpha=np.linspace(0.03, 0.99, 33), rho=0.5, seed=np.arange(25)
    )
    df_EP = df_EP.groupby(["alpha", "source", "rho"], as_index=False).mean()
    del df_EP["seed"]
    df_SE = run_experiments(run_SE, alpha=np.linspace(0.01, 1.0, 100), rho=0.5)
    df_BO = run_experiments(run_BO, alpha=np.linspace(0.01, 1.0, 100), rho=0.5)
    # concat and save
    df = pd.concat([df_EP, df_SE, df_BO], ignore_index=True, sort=False)
    csv_file = __file__.replace(".py", ".csv")
    logging.info(f"Saving {csv_file}")
    df.to_csv(csv_file, index=False)
    # plot
    plot_mse_curves()

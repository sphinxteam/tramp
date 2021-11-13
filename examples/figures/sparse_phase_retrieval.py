import logging
import numpy as np
import pandas as pd
from tramp.models import glm_generative, glm_state_evolution
from tramp.experiments import run_experiments, BayesOptimalScenario
from tramp.algos import EarlyStopping, CustomInit, StateEvolution
from tramp.algos.metrics import sign_symmetric_mse
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
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
    ax.legend(loc='lower left', fancybox=True, shadow=False)
    fig.tight_layout()
    # save figure
    pdf_file = __file__.replace(".py", ".pdf")
    logging.info(f"Saving {pdf_file}")
    plt.savefig(
        pdf_file, format='pdf', dpi=1000, bbox_inches="tight", pad_inches=0.1
    )


def run_EP(alpha, rho, seed):
    model = glm_generative(
        N=1000, alpha=alpha, ensemble_type="gaussian",
        prior_type="gauss_bernoulli", output_type="abs",
        prior_rho=rho, prior_mean=0.01
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    scenario.setup(seed)
    early = EarlyStopping(wait_increase=10, tol=1e-4)
    x_data = scenario.run_ep(max_iter=200, damping=0.1, callback = early)
    x_pred = x_data["x"]["r"]
    mse = sign_symmetric_mse(x_pred, scenario.x_true["x"])
    return dict(source="EP", v=mse)


def run_SE(alpha, rho):
    # analytical linear channel (Marcenko Pastur)
    model = glm_state_evolution(
        alpha=alpha, prior_type="gauss_bernoulli", output_type="abs",
        prior_rho=rho, prior_mean=0.01
    )
    # SE : uninformed initialization
    initializer = CustomInit(a_init=[("x", "bwd", 0.1)])
    early = EarlyStopping(wait_increase=10, tol=1e-4)
    se = StateEvolution(model)
    se.iterate(max_iter = 200, initializer=initializer, callback=early)
    x_data = se.get_variable_data(id="x")
    records = [dict(source="SE", v=x_data["v"])]
    # Bayes optimal : informed initialization
    initializer = CustomInit(a_init=[("x", "bwd", 10**3)])
    early = EarlyStopping(wait_increase=10, tol=1e-4)
    se = StateEvolution(model)
    se.iterate(max_iter = 200, initializer=initializer, callback=early)
    x_data = se.get_variable_data(id="x")
    records += [dict(source="BO", v=x_data["v"])]
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # run 25 instances of EP
    df_EP = run_experiments(
        run_EP, alpha=np.linspace(0.03, 1.2, 40), rho=0.6, seed=np.arange(25)
    )
    # avg over the 25 instances
    df_EP = df_EP.groupby(["alpha", "source", "rho"]).mean().reset_index()
    del df_EP["seed"]
    # run SE
    df_SE = run_experiments(
        run_SE, alpha=np.linspace(0.01, 1.2, 120), rho=0.6
    )
    # concat and save
    df = df_EP.append(df_SE, ignore_index=True)
    csv_file = __file__.replace(".py", ".csv")
    df.to_csv(csv_file, index=False)
    # plot
    plot_mse_curves()

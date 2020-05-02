from utils.plot.import_library_plot import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from utils.functions import save_object, load_object

from tramp.algos import CustomInit
from tramp.algos.metrics import mean_squared_error
from tramp.experiments import BayesOptimalScenario
from tramp.priors import GaussBernouilliPrior
from tramp.channels import GaussianChannel, LinearChannel, AbsChannel
from tramp.ensembles import GaussianEnsemble
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.algos import EarlyStopping


def plot_sparse_PR(dic, save_fig=False, block=False):
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = cm.get_cmap('plasma_r')
    tab_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
    tab_l1, tab_l2,  tab_l3 = [], [], []

    ind = np.where(np.array(dic['tab_mse_se_uni']) > 1e-3)[0][-1]
    ax.plot(dic['tab_alpha'][:ind+1], dic['tab_mse_se_uni'][:ind+1],
            color=tab_col[0], lw=1.75, label=r'SE')
    ax.plot([dic['tab_alpha'][ind], dic['tab_alpha'][ind]], [dic['tab_mse_se_uni'][ind], dic['tab_mse_se_uni'][ind+1]], ':',
            color=tab_col[0], lw=1.75)
    ax.plot(dic['tab_alpha'][ind+1:], dic['tab_mse_se_uni'][ind+1:],
            color=tab_col[0], lw=1.75)

    delta = 3
    ax.plot(dic['tab_alpha'][::delta], dic['tab_mse_ep'][::delta],
            'o', color=tab_col[1], label=r'EP')

    ax.plot(dic['tab_alpha'], dic['tab_mse_se_inf'], '--',
            color=tab_col[2], lw=1.75, label=r'Bayes opt.')

    """ Ticks   """
    ax.set_xlim([0, max(dic['tab_alpha'])])
    ax.set_ylim([-1e-3, max(dic['tab_mse_se_inf'])])

    """ Labels """
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'MSE')

    ax.legend(loc='upper right', fancybox=True,
              shadow=False, ncol=1)

    # Labels

    """ Save   """
    if save_fig:
        dir_fig = 'Figures/'
        os.makedirs(dir_fig) if not os.path.exists(dir_fig) else 0
        file_name = f'{dir_fig}/PR_rho={params["rho"]}_N={params["N"]}.pdf'
        plt.tight_layout()
        plt.savefig(file_name, format='pdf', dpi=1000,
                    bbox_inches="tight", pad_inches=0.1)

    """ Show   """
    if block:
        plt.show(block=False)
        input('Press enter to continue')
        plt.close()
    else:
        plt.close()


class Phase_retrieval():
    def __init__(self, params, seed=False):
        self.N = params['N']
        self.alpha = params['alpha']
        self.mean = 0.01
        self.M = int(self.alpha * self.N)
        self.rho = params['rho']
        self.model = self.build_model()
        self.scenario = self.build_scenario(seed)

    def build_model(self):
        prior = GaussBernouilliPrior(size=self.N, mean=self.mean, rho=self.rho)
        ensemble = GaussianEnsemble(self.M, self.N)
        W = ensemble.generate()
        model = prior @ V(id="x") @ LinearChannel(
            W=W, name='W') @ V(id="z") @ AbsChannel() @ O(id="y")
        model = model.to_model()
        return model

    def build_scenario(self, seed):
        scenario = BayesOptimalScenario(self.model, x_ids=["x"])
        scenario.setup(seed=seed)
        return scenario


def run_ep(scenario, settings, n_samples=10):
    callback = EarlyStopping(wait_increase=10)
    tab_mse = {'mse_ep': [], 'mse': []}

    # Average EP over n_samples #
    for i in range(n_samples):
        scenario.setup(seed=i)
        ep_x_data = scenario.run_ep(
            max_iter=settings['max_iter'], callback=callback, damping=settings['damping'])
        mse = mean_squared_error(scenario.x_pred['x'], scenario.x_true['x'])
        tab_mse['mse'].append(mse)
        tab_mse['mse_ep'].append(ep_x_data['x']['v'])
    mse, mse_ep = np.mean(tab_mse['mse']), np.mean(tab_mse['mse_ep'])

    print(f'mse ep:{mse_ep:.3e}')
    return mse


def run_se(scenario, settings):
    callback = EarlyStopping(wait_increase=10)

    # UNI-nformative Initialization #
    a_init = [("x", "bwd", 0.1)]
    initializer = CustomInit(a_init=a_init)
    data_se = scenario.run_se(
        max_iter=settings["max_iter"],
        damping=settings['damping'],
        initializer=initializer,
        callback=callback)
    mse_se_uni = data_se['x']['v']
    print(f'mse se uni:{mse_se_uni:.3e}')

    # INF-ormative Initialization #
    a_init = [("x", "bwd", 10**3)]
    initializer = CustomInit(a_init=a_init)
    data_se = scenario.run_se(
        max_iter=settings["max_iter"],
        damping=settings['damping'],
        initializer=initializer,
        callback=callback)
    mse_se_inf = data_se['x']['v']
    print(f'mse se inf:{mse_se_inf:.3e}')

    return mse_se_uni, mse_se_inf


def compute_mse_curve(params, settings, n_points=10, n_samples=1, seed=False, save_data=False):
    tab_alpha_ = np.linspace(0.0025, 1.2, n_points)
    dic = {key: [] for key in ['tab_alpha',
                               'tab_mse_se_inf',
                               'tab_mse_se_uni',
                               'tab_mse_ep']
           }
    for alpha in tab_alpha_:
        # Create TRAMP instance #
        print(f'\n alpha:{alpha}')
        params['alpha'] = alpha
        pr = Phase_retrieval(params, seed)
        scenario = pr.scenario
        # Run TRAMP EP ##
        mse_ep = run_ep(scenario, settings, n_samples=n_samples)
        # Run TRAMP SE ##
        mse_uni, mse_inf = run_se(scenario, settings)

        # Append data #
        dic['tab_alpha'].append(alpha)
        dic['tab_mse_se_inf'].append(mse_inf)
        dic['tab_mse_se_uni'].append(mse_uni)
        dic['tab_mse_ep'].append(mse_ep)

    if save_data:
        dir_data = 'Data'
        file_name = f'{dir_data}/PR_rho={params["rho"]:.2f}_N={params["N"]}.pkl'
        os.makedirs(dir_data) if not os.path.exists(dir_data) else 0
        save_object(dic, file_name)

    return dic


if __name__ == "__main__":
    ## Define parameters and settings ##
    params = {'N': 1000, 'rho': 0.6}
    settings_ep = {'damping': 0.3, 'max_iter': 200}
    settings_exp = {'n_points': 50, 'n_samples': 1}
    seed = True

    ## Compute and plot MSE curve as a function of alpha ##
    load_data = False
    save_data = True
    if not load_data:
        dic = compute_mse_curve(params, settings_ep,
                                n_points=settings_exp['n_points'], n_samples=settings_exp['n_samples'],
                                seed=seed, save_data=save_data)
    else:
        dic = load_object(
            f'Data/PR_rho={params["rho"]:.2f}_N={params["N"]}.pkl')

    ## Plot ##
    save_fig = False
    plot_sparse_PR(dic, block=True, save_fig=True)

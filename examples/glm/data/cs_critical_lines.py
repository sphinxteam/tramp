import logging
import numpy as np
from tramp.experiments import save_experiments, find_critical_alpha
from tramp.models import glm_state_evolution


def run_critical(a0, prior_rho, criterion):
    alpha = find_critical_alpha(
        id="x", a0=a0, mse_criterion=criterion,
        alpha_min=1e-5, alpha_max=2., alpha_tol=0.001,
        model_builder=glm_state_evolution,
        prior_type="gauss_bernoulli", output_type="gaussian",
        prior_rho=prior_rho, output_var=1e-11
    )
    return dict(alpha=alpha)


if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_critical, csv_file,
        a0=0, prior_rho=np.linspace(0.05, 0.95, 19), 
        criterion=["perfect"]
    )

import logging
import numpy as np
from tramp.models import glm_state_evolution
from tramp.experiments import save_experiments, run_state_evolution
from tramp.algos import CustomInit

def run_se(a0, alpha, prior_rho, prior_mean):
    model = glm_state_evolution(
        alpha=alpha, prior_type="gauss_bernoulli", output_type="modulus",
        prior_rho=prior_rho, prior_mean=prior_mean
    )
    a_init = [("x", "bwd", a0)]
    initializer = CustomInit(a_init=a_init)
    records = run_state_evolution(
        x_ids=["x"], model=model, max_iter=200, initializer=initializer
    )
    return records

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_se, csv_file,
        a0 = [0.1, 1000], prior_rho=[0.4, 0.6], prior_mean=0,
        alpha=np.linspace(0, 3, 151)[1:]
    )

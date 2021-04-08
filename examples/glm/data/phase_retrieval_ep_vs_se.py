import logging
import numpy as np
from tramp.models import glm_generative
from tramp.experiments import save_experiments, BayesOptimalScenario
from tramp.algos import EarlyStopping, EarlyStoppingEP

def run_phase_retrieval(N, alpha, prior_mean):
    model = glm_generative(
        N=N, alpha=alpha, ensemble_type="complex_gaussian", 
        prior_type="gauss_bernoulli", output_type="modulus",
        prior_mean=prior_mean, prior_rho=0.5 
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    early = EarlyStopping(wait_increase=10)
    records = scenario.run_all(
        metrics=["mse", "phase_mse"], 
        max_iter=200, damping=0.3, callback=early
    )
    return records

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_phase_retrieval, csv_file, 
        N=1000, 
        prior_mean=[0.01, 0.1, 1], alpha=np.linspace(0, 3, 151)[1:]
    )
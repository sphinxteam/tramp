import logging
import numpy as np
from tramp.models import glm_generative
from tramp.experiments import save_experiments, BayesOptimalScenario
from tramp.algos import EarlyStopping, EarlyStoppingEP

def run_cs(N, alpha, f, prior_rho):
    model = glm_generative(
        N=N, alpha=alpha, ensemble_type="random_feature",
        prior_type="gauss_bernoulli", output_type="gaussian",
        ensemble_f=f, prior_rho=prior_rho, output_var=1e-11
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    early = EarlyStopping()
    records = scenario.run_all(
        metrics=["mse"], max_iter=200, callback=early
    )
    return records

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_cs, csv_file, 
        N=1000, f=["abs", "sgn", "relu", "tanh"], 
        prior_rho=[0.25, 0.50, 0.75], alpha=np.linspace(0,1,50)[1:]
    )
import logging
import numpy as np
from tramp.models import glm_generative
from tramp.experiments import save_experiments, BayesOptimalScenario
from tramp.algos import EarlyStopping

def run_perceptron(N, alpha, p_pos):
    model = glm_generative(
        N=N, alpha=alpha, 
        ensemble_type="gaussian", prior_type="binary", output_type="sgn", 
        prior_p_pos=p_pos
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    early = EarlyStopping()
    records = scenario.run_all(max_iter=200, callback=early)
    return records

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_perceptron, csv_file, 
        N=1000, p_pos=[0.25, 0.50, 0.75], alpha=np.linspace(0, 2, 101)[1:]
    )
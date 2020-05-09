import logging
import numpy as np
from tramp.experiments import (
    save_experiments, find_critical_alpha, log_on_progress
)
from tramp.models import glm_state_evolution

def run_critical(a0, output_width, prior_p_pos, criterion):
    alpha = find_critical_alpha(
        id="x", a0=a0, mse_criterion=criterion,
        alpha_min=0.1, alpha_max=3., alpha_tol=0.001,
        model_builder=glm_state_evolution,
        prior_type="binary", output_type="door",
        output_width=output_width, prior_p_pos=prior_p_pos
    )
    return dict(alpha=alpha)

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_critical, csv_file,
        a0=[0.1, 1000], output_width=np.linspace(0.05, 1.95, 39),
        prior_p_pos=0.51,
        criterion=["random", "perfect"],
        on_progress=log_on_progress
    )

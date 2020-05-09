import logging
import numpy as np
from tramp.models import glm_state_evolution
from tramp.experiments import save_experiments, run_state_evolution
from tramp.algos import CustomInit, EarlyStopping

def run_se(a0, alpha, output_width, prior_p_pos):
    model = glm_state_evolution(
        alpha=alpha, prior_type="binary", output_type="door",
        output_width=output_width, prior_p_pos=prior_p_pos
    )
    a_init = [("x", "bwd", a0)]
    initializer = CustomInit(a_init=a_init)
    early = EarlyStopping(max_increase=0.1)
    records = run_state_evolution(
        x_ids=["x"], model=model,
        max_iter = 200, initializer=initializer,
        callback = early
    )
    return records

if __name__=="__main__":
    csv_file = __file__.replace(".py", ".csv")
    logging.basicConfig(level=logging.INFO)
    save_experiments(
        run_se, csv_file,
        #a0 = [0.1, 1000],
        a0 = 0.1, output_width=[0.5, 1.0, 1.5], prior_p_pos=0.51,
        alpha=np.linspace(0.5, 3.0, 51)[1:]
    )

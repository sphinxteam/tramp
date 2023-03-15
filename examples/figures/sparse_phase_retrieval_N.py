import logging
import argparse
import numpy as np
import pandas as pd
from tramp.models import glm_generative
from tramp.experiments import run_experiments, BayesOptimalScenario
from tramp.algos import EarlyStoppingEP
from tramp.algos.metrics import sign_symmetric_mse


def run_EP(N, alpha, rho):
    model = glm_generative(
        N=N, alpha=alpha, ensemble_type="gaussian",
        prior_type="gauss_bernoulli", output_type="abs",
        prior_rho=rho, prior_mean=0.01
    )
    scenario = BayesOptimalScenario(model, x_ids=["x"])
    mses = []
    # avg over 100 instances
    for _ in range(100):
        scenario.setup()
        callback = EarlyStoppingEP(tol=1e-4)
        x_data = scenario.run_ep(max_iter=200, damping=0.3, callback=callback)
        mse = sign_symmetric_mse(x_data["x"]["r"], scenario.x_true["x"])
        mses.append(mse)
    return dict(source="EP", v=np.mean(mses))


def merge_csv(N, step):
    csv_files = [
        f"sparse_phase_retrieval_N{N}_{start}:{step}.csv" 
        for start in range(step)
    ]
    df = pd.concat(
        [pd.read_csv(csv_file) for csv_file in csv_files],
        ignore_index=True, sort=False
    ).sort_values(by="alpha")
    csv_file = f"sparse_phase_retrieval_N{N}.csv"
    logging.info(f"Saving ${csv_file}")
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("start", type=int)
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    alphas = np.linspace(0.02, 1.18, 30)
    df = run_experiments(run_EP, N=args.N, alpha=alphas[args.start::args.step], rho=0.6)
    csv_file = f"sparse_phase_retrieval_N{args.N}_{args.start}:{args.step}.csv"
    logging.info(f"Saving ${csv_file}")
    df.to_csv(csv_file, index=False)

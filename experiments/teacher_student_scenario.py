import logging
import pandas as pd
from ..algos import ExpectationPropagation, StateEvolution
from ..algos import TrackErrors, TrackEvolution, EarlyStopping, JoinCallback
from ..models import Model
from ..algos.metrics import METRICS


class TeacherStudentScenario():
    """Implements teacher student scenario.

    Parameters
    ----------
    - model : Model instance
        Generative model
    - x_ids : ids of the variables to infer (signals)
    - y_ids : ids of the observed variables (measurements)
    """

    def __init__(self, model, x_ids=["x"], y_ids=["y"]):
        if not isinstance(model, Model):
            raise ValueError(f"{model} not a Model")
        for x_id in x_ids:
            if x_id not in model.variable_ids:
                raise ValueError(f"x_id = {x_id} not in model variable_ids")
        for y_id in y_ids:
            if y_id not in model.variable_ids:
                raise ValueError(f"y_id = {y_id} not in  model variable_ids")
        self.x_ids = x_ids
        self.y_ids = y_ids
        self.teacher = model

    def setup(self):
        # teacher generate data
        sample = self.teacher.sample()
        self.true_values = sample
        self.x_true = {x_id: sample[x_id] for x_id in self.x_ids}
        self.observations = {y_id: sample[y_id] for y_id in self.y_ids}
        # pass it to the student
        self.student = self.teacher.to_observed(self.observations)

    def infer(self, **kwargs):
        self.run_ep(**kwargs)
        self.run_se(**kwargs)
        self.compute_score(metrics=["mse", "overlap"])

    def run_se(self, **kwargs):
        se = StateEvolution(self.student)
        se.iterate(**kwargs)
        se_x_data = se.get_variables_data(self.x_ids)
        self.v_se = {x_id: data["v"] for x_id, data in se_x_data.items()}
        self.n_iter_se = se.n_iter

    def run_ep(self, **kwargs):
        ep = ExpectationPropagation(self.student)
        ep.iterate(**kwargs)
        ep_x_data = ep.get_variables_data(self.x_ids)
        self.x_pred = {x_id: data["r"] for x_id, data in ep_x_data.items()}
        self.v_ep = {x_id: data["v"] for x_id, data in ep_x_data.items()}
        self.n_iter_ep = ep.n_iter

    def ep_convergence(self, metrics, variables_damping=None):
        early = EarlyStopping(tol=1e-6, min_variance=1e-10)
        track = TrackErrors(true_values=self.x_true, metrics=metrics)
        evo = TrackEvolution()
        callback = JoinCallback([track, evo, early])
        try:
            self.run_ep(
                max_iter=250, callback=callback, check_decreasing=False,
                variables_damping=variables_damping
            )
        except Exception as e:
            logging.error(e)
        df = pd.merge(
            track.get_dataframe(), evo.get_dataframe(), on=["id", "iter"]
        )
        for y in ["v"] + metrics:
            df[y] = df[y].clip(0, 2)
        return df

    def compute_score(self, metrics=["mse"]):
        self.score = {
            metric: {
                x_id: METRICS.get(metric)(self.x_true[x_id], self.x_pred[x_id])
                for x_id in self.x_ids
            }
            for metric in metrics
        }

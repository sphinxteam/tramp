from ..algos.metrics import METRICS
from ..models import Model
from ..algos import TrackErrors, TrackEvolution, EarlyStopping, JoinCallback
from ..algos import ExpectationPropagation, StateEvolution
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class TeacherStudentScenario():
    """Implements teacher student scenario.

    Parameters
    ----------
    - teacher : Model instance or any object with a `.sample()` method.
        Generative teacher model
    - student : Model instance
        Generative student model
    - x_ids : ids of the variables to infer (signals)
    - y_ids : ids of the observed variables (measurements)
    """

    def __init__(self, teacher, student, x_ids=["x"], y_ids=["y"]):
        if not isinstance(student, Model):
            raise ValueError("student not a Model")
        try:
            sample = teacher.sample()
        except AttributeError:
            raise ValueError("teacher does not have a .sample() method")
        for x_id in x_ids:
            if x_id not in student.variable_ids:
                raise ValueError(f"x_id = {x_id} not in student variable_ids")
            if x_id not in sample:
                raise ValueError(f"x_id = {x_id} not in teacher variable_ids")
        for y_id in y_ids:
            if y_id not in student.variable_ids:
                raise ValueError(f"y_id = {y_id} not in  student variable_ids")
            if y_id not in sample:
                raise ValueError(f"y_id = {y_id} not in teacher variable_ids")
        self.x_ids = x_ids
        self.y_ids = y_ids
        self.teacher = teacher
        self.generative_student = student

    def setup(self):
        # teacher generate data
        sample = self.teacher.sample()
        self.true_values = sample
        self.x_true = {x_id: sample[x_id] for x_id in self.x_ids}
        self.observations = {y_id: sample[y_id] for y_id in self.y_ids}
        # pass it to the student
        self.student = self.generative_student.to_observed(self.observations)

    def run_all(self, source="ep,se", **kwargs):
        "Get mse values as estimated by EP or SE"
        self.setup()
        records = []
        if "se" in source:
            x_data = self.run_se(**kwargs)
            records += [
                dict(
                    source="se", x_id=x_id, v=x_data[x_id]["v"],
                    n_iter=x_data["n_iter"]
                ) for x_id in self.x_ids
            ]
        if "ep" in source:
            x_data = self.run_ep(**kwargs)
            records += [
                dict(
                    source="ep", x_id=x_id, v=x_data[x_id]["v"],
                    n_iter=x_data["n_iter"]
                ) for x_id in self.x_ids
            ]
            x_pred = {x_id: x_data["r"] for x_id in self.x_ids}
            score = self.compute_score(x_pred)
            records += [
                dict(
                    source="mse", x_id=x_id, v=score[x_id]["mse"]
                ) for x_id in self.x_ids
            ]
        return records

    def run_se(self, **algo_kwargs):
        se = StateEvolution(self.student)
        se.iterate(**algo_kwargs)
        x_data = se.get_variables_data(self.x_ids)
        x_data["n_iter"] = se.n_iter
        return x_data

    def run_ep(self, **algo_kwargs):
        ep = ExpectationPropagation(self.student)
        ep.iterate(**algo_kwargs)
        x_data = ep.get_variables_data(self.x_ids)
        x_data["n_iter"] = ep.n_iter
        return x_data

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
            logger.error(e)
        df = pd.merge(
            track.get_dataframe(), evo.get_dataframe(), on=["id", "iter"]
        )
        for y in ["v"] + metrics:
            df[y] = df[y].clip(0, 2)
        return df

    def se_convergence(self, metrics, variables_damping=None):
        early = EarlyStopping(tol=1e-6, min_variance=1e-10)
        evo = TrackEvolution()
        callback = JoinCallback([evo, early])
        try:
            self.run_se(
                max_iter=250, callback=callback, check_decreasing=False,
                variables_damping=variables_damping
            )
        except Exception as e:
            logger.error(e)
        df = evo.get_dataframe()
        for y in ["v"] + metrics:
            df[y] = df[y].clip(0, 2)
        return df

    def compute_score(self, x_pred, metrics=["mse"]):
        score = {
            x_id: {
                metric: f(self.x_true[x_id], x_pred[x_id])
                for metric, f in METRICS.items()
            }
            for x_id in self.x_ids
        }
        return score


class BayesOptimalScenario(TeacherStudentScenario):
    """Implements teacher student scenario in the Bayes Optimal setting.

    Parameters
    ----------
    - model : Model instance
        Same generative model for both teacher and student
    - x_ids : ids of the variables to infer (signals)
    - y_ids : ids of the observed variables (measurements)
    """

    def __init__(self, model, x_ids=["x"], y_ids=["y"]):
        super().__init__(teacher=model, student=model, x_ids=x_ids, y_ids=y_ids)


def run_state_evolution(x_ids, model, **algo_kwargs):
    """
    Run state evolution for a given model.

    Parameters
    ----------
    - x_ids : ids of the variables to infer (signals)
    - model : model that can be used in StateEvolution

    Returns
    -------
    - records : list
    """
    se = StateEvolution(model)
    se.iterate(**algo_kwargs)
    x_data = se.get_variables_data(ids=x_ids)
    records = [
        dict(x_id=x_id, v=x_data[x_id]["v"], n_iter=se.n_iter)
        for x_id in x_ids
    ]
    return records

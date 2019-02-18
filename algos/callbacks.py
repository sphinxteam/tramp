from ..base import ReprMixin
import logging
import numpy as np
import pandas as pd
from .metrics import METRICS


class Callback(ReprMixin):
    pass


class PassCallback(Callback):
    def __init__(self):
        self.repr_init()

    def __call__(self, algo,  i, max_iter):
        pass


class JoinCallback(Callback):
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.repr_init(pad="\t")

    def __call__(self, algo,  i, max_iter):
        stops = [
            callback(algo,  i, max_iter) for callback in self.callbacks
        ]
        stop = any(stops)
        return stop


class LogProgress(Callback):
    def __init__(self, ids="all", every=1):
        self.ids = ids
        self.every = every
        self.repr_init()

    def __call__(self, algo,  i, max_iter):
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            logging.info(f"iteration={i+1}/{max_iter}")
            for variable_id, data in variables_data.items():
                logging.info(f"id={variable_id} v={data['v']:.3f}")


class TrackEvolution(Callback):
    def __init__(self, ids="all", every=1):
        self.ids = ids
        self.every = every
        self.repr_init()
        self.records = []
        self.last = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            for variable_id, data in variables_data.items():
                record = dict(id=variable_id, v=data["v"], iter=i)
                self.records.append(record)

    def get_dataframe(self):
        return pd.DataFrame(self.records)


class TrackErrors(Callback):
    def __init__(self, true_values, metric="mse", every=1):
        self.ids = true_values.keys()
        self.metric = metric
        self.every = every
        self.repr_init()
        self.X_true = true_values
        self.compute_metric = METRICS.get(metric)
        self.errors = []

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.errors = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            X_pred = {
                variable_id: data["r"]
                for variable_id, data in variables_data.items()
            }
            errors = [{
                "id": id,
                self.metric: self.compute_metric(X_pred[id], self.X_true[id]),
                "iter": i
            } for id in self.ids]
            self.errors += errors

    def get_dataframe(self):
        return pd.DataFrame(self.errors)


class EarlyStopping(Callback):
    def __init__(self, ids="all", tol=1e-10, min_variance=1e-10):
        self.ids = ids
        self.tol = tol
        self.min_variance = min_variance
        self.repr_init()
        self.old_vs = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.old_vs = None
        variables_data = algo.get_variables_data(self.ids)
        new_vs = [data["v"] for variable_id, data in variables_data.items()]
        if any(v < self.min_variance for v in new_vs):
            logging.info(f"early stopping: min variance {min(new_vs)}")
            return True
        if any(np.isnan(v) for v in new_vs):
            logging.info(f"early stopping: nan values")
            return True
        if self.old_vs:
            tols = [
                np.abs(old_v - new_v)
                for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            logging.debug(f"tolerances max={max(tols):.2e} min={min(tols):.2e}")
            if max(tols) < self.tol:
                logging.info(f"early stopping: all tolerances are below tol={self.tol:.2e}")
                return True
        # for next iteration
        self.old_vs = new_vs

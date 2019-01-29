from ..base import ReprMixin
import logging
import numpy as np
from sklearn.metrics import mean_squared_error

METRICS = {
    "mse": mean_squared_error
}


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
    def __init__(self, every=1):
        self.every = every
        self.repr_init()

    def __call__(self, algo,  i, max_iter):
        if (i % self.every == 0):
            variables_data = algo.get_variables_data()
            logging.info(f"iteration={i+1}/{max_iter}")
            for data in variables_data:
                logging.info(f"variable={data['variable']} v={data['v']:.3f}")


class TrackEvolution(Callback):
    def __init__(self, every=1):
        self.every = every
        self.repr_init()
        self.records = []
        self.last = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data()
            for data in variables_data:
                record = dict(id=data["id"], v=data["v"], iter=i)
                self.records.append(record)


class TrackErrors(Callback):
    def __init__(self, true_values, metric="mse", every=1):
        self.metric = metric
        self.every = every
        self.repr_init()
        self.X_true = {
            data["id"]: data["X"] for data in true_values
        }
        self.compute_metric = METRICS.get(metric)
        self.errors = []

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.errors = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data()
            X_pred = {
                data["id"]: data["r"] for data in variables_data
            }
            errors = [
                {
                    "id": id,
                    self.metric: self.compute_metric(
                        X_pred[id], self.X_true[id]
                    ),
                    "iter": i
                }
                for id in self.X_true.keys()
            ]
            self.errors += errors


class EarlyStopping(Callback):
    def __init__(self, tol=1e-10, min_variance=1e-10):
        self.tol = tol
        self.min_variance = min_variance
        self.repr_init()
        self.old_vs = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.old_vs = None
        variables_data = algo.get_variables_data()
        new_vs = [data["v"] for data in variables_data]
        if any(v<self.min_variance for v in new_vs):
            logging.info(f"early stopping: min variance {min(new_vs)}")
            return True
        if any(np.isnan(v) for v in new_vs):
            logging.info(f"early stopping: nan values")
            return True
        if self.old_vs:
            tols = [
                np.mean((old_v - new_v) ** 2)
                for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            logging.debug(f"tolerances max={max(tols):.2e} min={min(tols):.2e}")
            if max(tols) < self.tol:
                logging.info(f"early stopping: all tolerances are below tol={self.tol:.2e}")
                return True
        # for next iteration
        self.old_vs = new_vs

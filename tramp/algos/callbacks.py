"""Callbacks for ExpectationPropagation and StateEvolution algorithms."""
from .metrics import METRICS
import pandas as pd
import numpy as np
from ..base import ReprMixin
import logging
logger = logging.getLogger(__name__)


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
            logger.info(f"iteration={i+1}/{max_iter}")
            for variable_id, data in variables_data.items():
                logger.info(f"id={variable_id} v={data['v']:.3f}")


class TrackMessages(Callback):
    def __init__(self, keys=["a", "n_iter", "direction"]):
        self.keys = keys
        self.records = []

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        self.records += algo.get_edges_data(self.keys)

    def get_dataframe(self):
        return pd.DataFrame(self.records)


class TrackObjective(Callback):
    def __init__(self):
        self.edge_records = []
        self.node_records = []
        self.model_records = []

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        algo.update_objective()
        # model
        model_record = dict(A=algo.A_model, n_iter=algo.n_iter)
        self.model_records.append(model_record)
        # edges
        self.edge_records += algo.get_edges_data(["A", "n_iter", "direction"])
        # nodes
        self.node_records += algo.get_nodes_data(["A", "n_iter"])

    def get_dataframe(self):
        edge_df = pd.DataFrame(self.edge_records)
        node_df = pd.DataFrame(self.node_records)
        model_df = pd.DataFrame(self.model_records)
        return edge_df, node_df, model_df


class TrackEvolution(Callback):
    def __init__(self, ids="all", every=1, verbose=False):
        self.ids = ids
        self.every = every
        self.repr_init()
        self.records = []
        self.verbose = verbose

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            for variable_id, data in variables_data.items():
                record = dict(id=variable_id, v=data["v"], iter=i)
                self.records.append(record)
                if self.verbose:
                    print(record)

    def get_dataframe(self):
        return pd.DataFrame(self.records)


class TrackEstimate(Callback):
    def __init__(self, ids="all", every=1):
        self.ids = ids
        self.every = every
        self.repr_init()
        self.records = []

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            for variable_id, data in variables_data.items():
                record = dict(id=variable_id, r=data["r"], iter=i)
                self.records.append(record)

    def get_dataframe(self):
        return pd.DataFrame(self.records)


class TrackErrors(Callback):
    def __init__(self, true_values, metrics=["mse"], every=1, verbose=False):
        self.ids = true_values.keys()
        self.metrics = metrics
        self.every = every
        self.repr_init()
        self.X_true = true_values
        self.errors = []
        self.verbose = verbose

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.errors = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            X_pred = {
                variable_id: data["r"]
                for variable_id, data in variables_data.items()
            }
            errors = []
            for id in self.ids:
                error = dict(id=id, iter=i)
                for metric in self.metrics:
                    func = METRICS.get(metric)
                    error[metric] = func(X_pred[id], self.X_true[id])
                errors.append(error)
            if self.verbose:
                print(errors)
            self.errors += errors

    def get_dataframe(self):
        return pd.DataFrame(self.errors)


class TrackOverlaps(Callback):
    def __init__(self, true_values, ids="all", every=1, verbose=False):
        self.ids = ids
        self.every = every
        self.repr_init()
        self.X_true = true_values
        self.records = []
        self.verbose = verbose

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.records = []
        if (i % self.every == 0):
            variables_data = algo.get_variables_data(self.ids)
            for variable_id, data in variables_data.items():
                m = 1/self.X_true[variable_id].shape[0] * \
                    (data['r'].T).dot(self.X_true[variable_id])
                q = 1/self.X_true[variable_id].shape[0] * \
                    (data['r'].T).dot(data['r'])
                Q = 1/self.X_true[variable_id].shape[0] * \
                    (self.X_true[variable_id].T).dot(self.X_true[variable_id])
                record = dict(id=variable_id, m=m, q=q, Q=Q, iter=i)
                self.records.append(record)
                if self.verbose:
                    print(record)

    def get_dataframe(self):
        return pd.DataFrame(self.records)


class EarlyStopping(Callback):
    def __init__(self, ids="all", tol=1e-6, min_variance=-1,
                 wait_increase=5, max_increase=0.2):
        self.ids = ids
        self.tol = tol
        self.min_variance = min_variance
        self.wait_increase = wait_increase
        self.max_increase = max_increase
        self.repr_init()
        self.old_vs = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.old_vs = None
        variables_data = algo.get_variables_data(self.ids)
        new_vs = [data["v"] for variable_id, data in variables_data.items()]
        if any(v < self.min_variance for v in new_vs):
            logger.info(f"early stopping min variance {min(new_vs)}")
            return True
        if any(np.isnan(v) for v in new_vs):
            logger.warning("early stopping nan values")
            logger.info("restoring old message dag")
            algo.reset_message_dag(self.old_message_dag)
            return True
        if self.old_vs:
            tols = [
                np.abs(old_v - new_v)
                for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            if max(tols) < self.tol:
                logger.info(
                    "early stopping all tolerances (on v) are "
                    f"below tol={self.tol:.2e}"
                )
                return True
            increase = [
                new_v - old_v for old_v, new_v in zip(self.old_vs, new_vs)
            ]
            if i > self.wait_increase and max(increase) > self.max_increase:
                logger.info(
                    f"increase={max(increase)} above "
                    f"max_increase={self.max_increase:.2e}"
                )
                logger.info("restoring old message dag")
                algo.reset_message_dag(self.old_message_dag)
                return True
        # for next iteration
        self.old_vs = new_vs
        self.old_message_dag = algo.message_dag.copy()


def norm(x):
    return np.sqrt(np.mean(x**2))


class EarlyStoppingEP(Callback):
    def __init__(self, ids="all", tol=1e-6, wait_increase=5, max_increase=0.2):
        self.ids = ids
        self.tol = tol
        self.wait_increase = wait_increase
        self.max_increase = max_increase
        self.repr_init()
        self.old_rs = None

    def __call__(self, algo,  i, max_iter):
        if (i == 0):
            self.old_rs = None
        variables_data = algo.get_variables_data(self.ids)
        new_rs = [data["r"] for variable_id, data in variables_data.items()]
        if self.old_rs:
            tols = [
                norm(new_r - old_r)/norm(new_r)
                for old_r, new_r in zip(self.old_rs, new_rs)
            ]
            if max(tols) < self.tol:
                logger.info(
                    "early stopping all tolerances (on r) are "
                    f"below tol={self.tol:.2e}"
                )
                return True
            increase = tols
            if i > self.wait_increase and max(increase) > self.max_increase:
                logger.info(
                    f"increase={max(increase)} above "
                    f"max_increase={self.max_increase:.2e}"
                )
                logger.info("restoring old message dag")
                algo.reset_message_dag(self.old_message_dag)
                return True
        # for next iteration
        self.old_rs = new_rs
        self.old_message_dag = algo.message_dag.copy()

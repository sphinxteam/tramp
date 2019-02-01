from ..base import Variable, Factor, Model
from .dag_algebra import ModelDAG
import numpy as np
import networkx as nx


def to_list(X):
    if not isinstance(X, tuple):
        X = (X,)
    return list(X)


def check_variable_ids(variables):
    for i, variable in enumerate(variables):
        if variable.id is None:
            raise ValueError("missing id for the i={i} {variable} ")
    n_unique = len(set(variables))
    n_variables = len(variables)
    if n_unique != n_variables:
        raise ValueError(f"having {n_unique} ids but {n_variables} variables")


class DAGModel(Model):
    def __init__(self, model_dag):
        if not isinstance(model_dag, ModelDAG):
            raise TypeError(f"model_dag {model_dag} is not a ModelDAG")
        self.model_dag = model_dag
        self.dag = model_dag.dag.copy()
        self.forward_ordering = nx.topological_sort(self.dag)
        self.backward_ordering = list(reversed(self.forward_ordering))
        self.variables = [
            node for node in self.forward_ordering
            if isinstance(node, Variable)
        ]
        check_variable_ids(self.variables)
        self.n_variables = len(self.variables)
        self.factors = [
            node for node in self.forward_ordering
            if isinstance(node, Factor)
        ]
        self.n_factors = len(self.factors)
        self.init_shapes()
        self.second_moment()
        nx.freeze(self.dag)

    def daft(self, layout=None):
        self.model_dag.daft(layout)

    def sample(self):
        "Forward sampling of the model"
        sample_dag = nx.DiGraph()
        sample_dag.add_edges_from(self.dag.edges())
        for factor in self.factors:
            prev_variables = sample_dag.predecessors(factor)
            next_variables = sample_dag.successors(factor)
            X_prev = [
                sample_dag.node[variable]["X"]
                for variable in prev_variables
            ]
            X_next = factor.sample(*X_prev)
            X_next = to_list(X_next)
            for X, variable in zip(X_next, next_variables):
                sample_dag.node[variable].update(
                    X=X, shape=X.shape, id=variable.id
                )
        Xs = [sample_dag.node[variable] for variable in self.variables]
        return Xs

    def init_shapes(self):
        "Compute variable shapes inplace (shape attribute of variable node)"
        for factor in self.factors:
            prev_variables = self.dag.predecessors(factor)
            next_variables = self.dag.successors(factor)
            X_prev = [
                np.ones(self.dag.node[variable]["shape"])
                for variable in prev_variables
            ]
            X_next = factor.sample(*X_prev)
            X_next = to_list(X_next)
            for X, variable in zip(X_next, next_variables):
                self.dag.node[variable].update(shape=X.shape)

    def second_moment(self):
        "Compute second_moment inplace (tau attribute of variable node)"
        for factor in self.factors:
            if factor.n_next:
                prev_variables = self.dag.predecessors(factor)
                next_variables = self.dag.successors(factor)
                tau_prev = [
                    self.dag.node[variable]["tau"]
                    for variable in prev_variables
                ]
                tau_next = factor.second_moment(*tau_prev)
                tau_next = to_list(tau_next)
                for tau, variable in zip(tau_next, next_variables):
                    self.dag.node[variable].update(tau=tau)

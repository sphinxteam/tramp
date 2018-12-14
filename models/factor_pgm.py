from ..base import Variable, BridgeVariable, FinalVariable
from ..base import Factor, Prior, Likelihood, Channel
from .dag_model import DAGModel, to_list
from .factor_algebra import FactorDAG, PlaceHolder
import networkx as nx
import logging


def compute_augmented_dag(factor_dag):
    augmented_dag = nx.DiGraph()
    augmented_dag.add_edges_from(factor_dag.edges())
    variable_order = 0
    for factor in nx.topological_sort(factor_dag):
        if not isinstance(factor, PlaceHolder):
            in_edges = augmented_dag.in_edges(factor, data=True)
            out_edges = augmented_dag.out_edges(factor, data=True)
            X_prev = [data["X"] for source, target, data in in_edges]
            X_next = factor.sample(*X_prev)
            X_next = to_list(X_next)
            for X, (source, target, data) in zip(X_next, out_edges):
                data.update(X=X, shape=X.shape, id=variable_order)
                variable_order += 1
    for source, target, data in augmented_dag.edges(data=True):
        del data["X"]
    return augmented_dag


class FactorPGM(DAGModel):
    def __init__(self, factor_dag):
        if not isinstance(factor_dag, FactorDAG):
            raise TypeError(f"factor_dag {factor_dag} is not a FactorDAG")
        for node in factor_dag._roots_ph:
            raise ValueError(f"root node {node} not a prior")
        self.factor_dag = factor_dag
        augmented_dag = compute_augmented_dag(factor_dag.factor_dag)
        model_dag = self._build_model_dag(augmented_dag)
        DAGModel.__init__(self, model_dag)


    def _build_model_dag(self, augmented_dag):
        dag = nx.DiGraph()
        for factor in augmented_dag.nodes():
            if not isinstance(factor, PlaceHolder):
                dag.add_node(factor, type="factor")
        for source, target, data in augmented_dag.edges(data=True):
            if isinstance(target, PlaceHolder):
                variable = FinalVariable(shape=data['shape'], id=data['id'])
            else:
                variable = BridgeVariable(shape=data['shape'], id=data['id'])
            dag.add_node(variable, type="variable")
            dag.add_edge(source, variable, type="factor_to_variable", shape=variable.shape)
            if not isinstance(target, PlaceHolder):
                dag.add_edge(variable, target, type="variable_to_factor", shape=variable.shape)
        return dag

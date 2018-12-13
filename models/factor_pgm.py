from ..base import Variable, BridgeVariable, FinalVariable
from ..base import Factor, Prior, Likelihood, Channel
from .dag_model import DAGModel, to_list
import networkx as nx
import logging

class PlaceHolder(Factor):
    pass

def compute_augmented_dag(factor_dag):
    augmented_dag = nx.DiGraph()
    augmented_dag.add_edges_from(factor_dag.edges())
    variable_order = 0
    for factor in nx.topological_sort(factor_dag):
        in_edges = augmented_dag.in_edges(factor, data=True)
        out_edges = augmented_dag.out_edges(factor, data=True)
        X_prev = [data["X"] for source, target, data in in_edges]
        X_next = factor.sample(*X_prev)
        X_next = to_list(X_next)
        # add PlaceHolder for leaf channel nodes (one for each output)
        leaf_node = not factor_dag.successors(factor)
        if isinstance(factor, Likelihood):
            pass
        elif leaf_node and isinstance(factor, Channel):
            for X in X_next:
                augmented_dag.add_edge(
                    factor, PlaceHolder(),
                    X=X, shape=X.shape, id=variable_order
                )
                variable_order += 1
        else:
            if not len(X_next)==len(out_edges):
                raise ValueError(f"factor {factor} X_next:{len(X_next)} out_edges:{len(out_edges)}")
            for X, (source, target, data) in zip(X_next, out_edges):
                data.update(X=X, shape=X.shape, id=variable_order)
                variable_order += 1
    for source, target, data in augmented_dag.edges(data=True):
        del data["X"]
    return augmented_dag

def check_factor_dag(factor_dag):
    if not nx.is_directed_acyclic_graph(factor_dag):
        raise ValueError(f"factor_dag {factor_dag} not a DAG")
    for node in factor_dag.nodes():
        if not isinstance(node, Factor):
            raise ValueError(f"node {node} not a Factor")
        leaf_node = not factor_dag.successors(node)
        n_prev = len(factor_dag.predecessors(node))
        n_next = len(factor_dag.successors(node))
        if n_prev != node.n_prev:
            raise ValueError(f"node {node} has {n_prev} predecessors but should have {node.n_prev}")
        if not leaf_node and n_next != node.n_next:
            raise ValueError(f"node {node} has {n_next} successors but should have {node.n_next}")


class FactorPGM(DAGModel):
    def __init__(self, factor_dag):
        check_factor_dag(factor_dag)
        self.factor_dag = factor_dag
        augmented_dag = compute_augmented_dag(factor_dag)
        model_dag = self._build_model_dag(augmented_dag)
        DAGModel.__init__(self, model_dag)

    def _build_model_dag(self, augmented_dag):
        dag = nx.DiGraph()
        for factor in augmented_dag:
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

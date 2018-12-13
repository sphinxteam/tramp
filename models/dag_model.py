from ..base import Variable, BridgeVariable, FinalVariable
from ..base import Factor, Prior, Likelihood, Channel, Model
from .dag_layout import Layout
import networkx as nx
import logging


def to_list(X):
    if not isinstance(X, tuple):
        X = (X,)
    return list(X)


def check_model_dag(model_dag):
    if not nx.is_directed_acyclic_graph(model_dag):
        raise ValueError(f"model_dag {model_dag} not a DAG")
    for node in model_dag.nodes():
        if not (isinstance(node, Factor) or isinstance(node, Variable)):
            raise ValueError(f"node {node} should be Factor or Variable")
        predecessors = model_dag.predecessors(node)
        successors = model_dag.successors(node)
        n_prev = len(predecessors)
        n_next = len(successors)
        if n_prev != node.n_prev:
            raise ValueError(f"node {node} has {n_prev} predecessors but should have {node.n_prev}")
        if n_next != node.n_next:
            raise ValueError(f"node {node} has {n_prev} successors but should have {node.n_prev}")
        opposite_class = Factor if isinstance(node, Variable) else Variable
        for predecessor in predecessors:
            if not isinstance(predecessor, opposite_class):
                raise ValueError(f"predecessor {predecessor} of {node} not a {opposite_class}")
        for successor in successors:
            if not isinstance(successor, opposite_class):
                raise ValueError(f"successor {successor} of {node} not a {opposite_class}")


class DAGModel(Model):
    def __init__(self, model_dag):
        check_model_dag(model_dag)
        self.model_dag = model_dag
        self.forward_ordering = nx.topological_sort(model_dag)
        self.backward_ordering = list(reversed(self.forward_ordering))
        self.variables = [
            node for node in self.forward_ordering
            if isinstance(node, Variable)
        ]
        self.n_variables = len(self.variables)
        self.factors = [
            node for node in self.forward_ordering
            if isinstance(node, Factor)
        ]
        self.n_factors = len(self.factors)
        nx.freeze(self.model_dag)

    def __repr__(self):
        return f"DAGModel(n_factors={self.n_factors}, n_variables={self.n_variables})"

    def sample(self):
        sample_dag = nx.DiGraph()
        sample_dag.add_edges_from(self.model_dag.edges())
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
                sample_dag.node[variable].update(X=X)
        Xs = [
            dict(
                id=variable.id, variable=variable,
                X=sample_dag.node[variable]["X"]
            )
            for variable in self.variables
        ]
        return Xs

    def second_moment(self):
        "Compute second_moment inplace (tau attribute of variable node)"
        for factor in self.factors:
            if factor.n_next:
                prev_variables = self.model_dag.predecessors(factor)
                next_variables = self.model_dag.successors(factor)
                tau_prev = [
                    self.model_dag.node[variable]["tau"]
                    for variable in prev_variables
                ]
                tau_next = factor.second_moment(*tau_prev)
                tau_next = to_list(tau_next)
                for tau, variable in zip(tau_next, next_variables):
                    self.model_dag.node[variable].update(tau=tau)

    def daft(self, layout=None):
        layout = layout or Layout()
        layout.compute_dag(self.model_dag)
        from matplotlib import rc
        rc("font", family="serif", size=12)
        rc("text", usetex=True)
        import daft
        pgm = daft.PGM([layout.Lx, layout.Ly], origin=[0, 0])
        idx_Y = 0
        N = len(self.forward_ordering)
        for l, node in enumerate(self.forward_ordering):
            x = layout.dag.node[node]["x"]
            y = layout.dag.node[node]["y"]
            fixed = isinstance(node, Factor)
            pgm.add_node(daft.Node(l, node.math(), x, y, fixed=fixed))
            if isinstance(node, Likelihood):
                label = r"$Y_{}$".format(idx_Y)
                pgm.add_node(daft.Node(N + idx_Y, label, x + layout.dx, y, observed=True))
                pgm.add_edge(l, N + idx_Y)
                idx_Y += 1
        for source, target in self.model_dag.edges():
            l_source = self.forward_ordering.index(source)
            l_target = self.forward_ordering.index(target)
            pgm.add_edge(l_source, l_target)
        pgm.render()

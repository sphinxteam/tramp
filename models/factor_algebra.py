from ..base import Factor
from .dag_layout import Layout
import networkx as nx


class PlaceHolder(Factor):
    def __init__(self):
        self.repr_init()
    
    def math(self):
        return r"$\emptyset$"


class RootPlaceHolder(PlaceHolder):
    n_prev = 0
    n_next = 1


class LeafPlaceHolder(PlaceHolder):
    n_prev = 1
    n_next = 0


def check_factor_dag(factor_dag):
    if not nx.is_directed_acyclic_graph(factor_dag):
        raise ValueError(f"factor_dag {factor_dag} not a DAG")
    for node in factor_dag.nodes():
        if not isinstance(node, Factor):
            raise ValueError(f"node {node} not a Factor")
        n_prev = len(factor_dag.predecessors(node))
        n_next = len(factor_dag.successors(node))
        if n_prev != node.n_prev:
            raise ValueError(f"node {node} has {n_prev} predecessors but should have {node.n_prev}")
        if n_next != node.n_next:
            raise ValueError(f"node {node} has {n_next} successors but should have {node.n_next}")


def to_factor_dag(factor):
    if not isinstance(factor, Factor):
        raise ValueError(f"factor {factor} not a Factor")
    dag = nx.DiGraph()
    dag.add_node(factor)
    for _ in range(factor.n_next):
        dag.add_edge(factor, LeafPlaceHolder())
    for _ in range(factor.n_prev):
        dag.add_edge(RootPlaceHolder(), factor)
    return dag


class FactorDAG():
    def __init__(self, factor_dag):
        if isinstance(factor_dag, Factor):
            factor_dag = to_factor_dag(factor_dag)
        check_factor_dag(factor_dag)
        self.factor_dag = factor_dag
        factors = nx.topological_sort(factor_dag)
        self._leafs_ph = [
            factor for factor in factors
            if isinstance(factor, LeafPlaceHolder)
        ]
        self._roots_ph = [
            factor for factor in factors
            if isinstance(factor, RootPlaceHolder)
        ]

    def __add__(self, other):
        if not isinstance(other, FactorDAG):
            other = FactorDAG(other)
        factor_dag = nx.DiGraph()
        factor_dag.add_edges_from(self.factor_dag.edges())
        factor_dag.add_edges_from(other.factor_dag.edges())
        return FactorDAG(factor_dag)

    def __matmul__(self, other):
        if not isinstance(other, FactorDAG):
            other = FactorDAG(other)
        factor_dag = nx.DiGraph()
        factor_dag.add_edges_from(self.factor_dag.edges())
        factor_dag.add_edges_from(other.factor_dag.edges())
        # dag surgery
        for leaf, root in zip(self._leafs_ph, other._roots_ph):
            leaf_predecessors = self.factor_dag.predecessors(leaf)
            root_successors = other.factor_dag.successors(root)
            assert len(leaf_predecessors) == 1
            assert len(root_successors) == 1
            prev_factor = leaf_predecessors[0]
            next_factor = root_successors[0]
            factor_dag.remove_node(leaf)
            factor_dag.remove_node(root)
            factor_dag.add_edge(prev_factor, next_factor)
        return FactorDAG(factor_dag)

    def daft(self, layout=None):
        layout = layout or Layout()
        layout.compute_dag(self.factor_dag)
        from matplotlib import rc
        rc("font", family="serif", size=12)
        rc("text", usetex=True)
        import daft
        pgm = daft.PGM([layout.Lx, layout.Ly], origin=[0, 0])
        factors = nx.topological_sort(self.factor_dag)
        for l, node in enumerate(factors):
            x = layout.dag.node[node]["x"]
            y = layout.dag.node[node]["y"]
            pgm.add_node(daft.Node(l, node.math(), x, y, fixed=True))
        for source, target in self.factor_dag.edges():
            l_source = factors.index(source)
            l_target = factors.index(target)
            pgm.add_edge(l_source, l_target)
        pgm.render()

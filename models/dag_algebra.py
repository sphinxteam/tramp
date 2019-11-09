from ..base import Variable, Factor, ReprMixin
from ..likelihoods.base_likelihood import Likelihood
from ..priors.base_prior import Prior
from ..variables import SISOVariable, SILeafVariable
from ..channels import (
    GaussianChannel,
    AbsChannel, AsymmetricAbsChannel, SgnChannel, ReluChannel,
    HardTanhChannel, HardSigmoidChannel, SymmetricDoorChannel,
    ModulusChannel
)
from ..likelihoods import (
    GaussianLikelihood,
    AbsLikelihood, AsymmetricAbsLikelihood, SgnLikelihood, ReluLikelihood,
    HardTanhLikelihood, HardSigmoidLikelihood, SymmetricDoorLikelihood,
    ModulusLikelihood
)
from .dag_layout import Layout
import networkx as nx


def channel2likelihood(channel, y, y_name):
    if isinstance(channel, GaussianChannel):
        return GaussianLikelihood(y=y, y_name=y_name, var=channel.var)
    if isinstance(channel, AbsChannel):
        return AbsLikelihood(y=y, y_name=y_name)
    if isinstance(channel, AsymmetricAbsChannel):
        return AsymmetricAbsLikelihood(y=y, y_name=y_name, shift=channel.shift)
    if isinstance(channel, SgnChannel):
        return SgnLikelihood(y=y, y_name=y_name)
    if isinstance(channel, ReluChannel):
        return ReluLikelihood(y=y, y_name=y_name)
    if isinstance(channel, HardTanhChannel):
        return HardTanhLikelihood(y=y, y_name=y_name)
    if isinstance(channel, HardSigmoidChannel):
        return HardSigmoidLikelihood(y=y, y_name=y_name)
    if isinstance(channel, SymmetricDoorChannel):
        return SymmetricDoorLikelihood(y=y, y_name=y_name, width=channel.width)
    if isinstance(channel, ModulusChannel):
        return ModulusLikelihood(y=y, y_name=y_name)
    raise NotImplementedError(f"cannot convert {channel} to likelihood")


class PlaceHolder(ReprMixin):
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


def check_dag(dag):
    if not isinstance(dag, nx.DiGraph) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f"dag {dag} not a DAG")
    for node in dag.nodes():
        predecessors = dag.predecessors(node)
        successors = dag.successors(node)
        n_prev = len(predecessors)
        n_next = len(successors)
        if n_prev != node.n_prev:
            raise ValueError(
                f"node {node} has {n_prev} predecessors "
                f"but should have {node.n_prev}"
            )
        if n_next != node.n_next:
            raise ValueError(
                f"node {node} has {n_prev} successors "
                f"but should have {node.n_prev}"
            )


def to_dag(node):
    dag = nx.DiGraph()
    dag.add_node(node)
    for _ in range(node.n_next):
        dag.add_edge(node, LeafPlaceHolder())
    for _ in range(node.n_prev):
        dag.add_edge(RootPlaceHolder(), node)
    return dag


class DAG():
    def __init__(self, dag):
        if not isinstance(dag, nx.DiGraph):
            dag = to_dag(dag)
        check_dag(dag)
        self.dag = dag
        nodes = nx.topological_sort(dag)
        self._leafs_ph = [
            node for node in nodes
            if isinstance(node, LeafPlaceHolder)
        ]
        self._roots_ph = [
            node for node in nodes
            if isinstance(node, RootPlaceHolder)
        ]

    def __add__(self, other):
        if not isinstance(other, DAG):
            other = DAG(other)
        dag = nx.DiGraph()
        dag.add_edges_from(self.dag.edges())
        dag.add_edges_from(other.dag.edges())
        return DAG(dag)

    def __matmul__(self, other):
        if not isinstance(other, DAG):
            other = DAG(other)
        dag = nx.DiGraph()
        dag.add_edges_from(self.dag.edges())
        dag.add_edges_from(other.dag.edges())
        # dag surgery
        for leaf, root in zip(self._leafs_ph, other._roots_ph):
            leaf_predecessors = self.dag.predecessors(leaf)
            root_successors = other.dag.successors(root)
            assert len(leaf_predecessors) == 1
            assert len(root_successors) == 1
            prev_factor = leaf_predecessors[0]
            next_factor = root_successors[0]
            dag.remove_node(leaf)
            dag.remove_node(root)
            dag.add_edge(prev_factor, next_factor)
        return DAG(dag)

    def to_factor_dag(self):
        return FactorDAG(self.dag)

    def to_model_dag(self):
        return ModelDAG(self.dag)

    def to_model(self):
        from .base_model import Model
        return Model(self.to_model_dag())

    def plot(self, layout=None, show_observed=False):
        layout = layout or Layout()
        layout.compute_dag(self.dag)
        from matplotlib import rc
        rc("font", family="serif", size=12)
        rc("text", usetex=True)
        import daft
        pgm = daft.PGM([layout.Lx, layout.Ly], origin=[0, 0])
        nodes = nx.topological_sort(self.dag)
        n_nodes = len(nodes)
        id_obs = 0
        for l_node, node in enumerate(nodes):
            x = layout.dag.node[node]["x"]
            y = layout.dag.node[node]["y"]
            fixed = isinstance(node, Factor) or isinstance(node, PlaceHolder)
            pgm.add_node(daft.Node(
                l_node, node.math(), x, y, fixed=fixed
            ))
            if isinstance(node, Likelihood):
                l_obs = n_nodes + id_obs
                pgm.add_node(daft.Node(
                    l_obs, node.y_name, x + layout.dx, y, observed=True
                ))
                pgm.add_edge(l_node, l_obs)
                id_obs += 1
        for source, target in self.dag.edges():
            l_source = nodes.index(source)
            l_target = nodes.index(target)
            pgm.add_edge(l_source, l_target)
        pgm.render()


def check_factor_dag(dag):
    if not isinstance(dag, nx.DiGraph) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f"dag {dag} not a DAG")
    for node in dag.nodes():
        if not (isinstance(node, Factor) or isinstance(node, PlaceHolder)):
            raise ValueError(f"node {node} must be a Factor or PlaceHolder")


class FactorDAG(DAG):
    def __init__(self, dag):
        if isinstance(dag, Variable):
            raise ValueError(f"Cannot convert variable {dag} to a FactorDAG")
        elif isinstance(dag, Factor):
            dag = to_dag(dag)
        check_factor_dag(dag)
        super().__init__(dag)

    def to_model_dag(self):
        if self._roots_ph:
            raise ValueError(
                "cannot convert FactorDAG -> ModelDAG: "
                f"there are {len(self._roots_ph)} RootPlaceHolders"
            )
        dag = nx.DiGraph()
        id_x = id_y = 0
        for source, target in self.dag.edges():
            assert isinstance(source, Factor)
            if isinstance(target, PlaceHolder):
                variable = SILeafVariable(id=f"y_{id_y}")
                id_y += 1
            else:
                variable = SISOVariable(id=f"x_{id_x}")
                id_x += 1
            dag.add_edge(source, variable, type="factor_to_variable")
            if not isinstance(target, PlaceHolder):
                dag.add_edge(variable, target, type="variable_to_factor")
        return ModelDAG(dag)


def check_model_dag(dag):
    if not isinstance(dag, nx.DiGraph) or not nx.is_directed_acyclic_graph(dag):
        raise ValueError(f"dag {dag} not a DAG")
    for node in dag.nodes():
        if not (isinstance(node, Factor) or isinstance(node, Variable)):
            raise ValueError(f"node {node} should be a Factor or Variable")
        opposite_class = Factor if isinstance(node, Variable) else Variable
        for predecessor in dag.predecessors(node):
            if not isinstance(predecessor, opposite_class):
                raise ValueError(
                    f"predecessor {predecessor} of {node} "
                    f"must be a {opposite_class}"
                )
        for successor in dag.successors(node):
            if not isinstance(successor, opposite_class):
                raise ValueError(
                    f"successor {successor} of {node} "
                    f"must be a {opposite_class}"
                )


class ModelDAG(DAG):
    def __init__(self, dag):
        if isinstance(dag, Variable) or isinstance(dag, Factor):
            dag = to_dag(dag)
        check_model_dag(dag)
        super().__init__(dag)

    def to_observed(self, observations):
        """ModelDAG with observed variables.

        Parameters
        ----------
        observations: dict of arrays
            observations = {id: observation}
        """
        observed_ids = observations.keys()

        def is_observed(node):
            if not isinstance(node, Variable):
                return False
            return (node.id in observed_ids)

        def is_likelihood(node):
            if not isinstance(node, Factor):
                return False
            successors = self.dag.successors(node)
            return any(variable.id in observed_ids for variable in successors)

        def get_observation_ids(node):
            if not isinstance(node, Factor):
                raise ValueError(f"{node} not a Factor")
            successors = self.dag.successors(node)
            return [
                variable.id
                for variable in successors if variable.id in observed_ids
            ]

        dag = nx.DiGraph()
        for source, target in self.dag.edges():
            if is_observed(target):
                if target.n_next != 0:
                    raise ValueError("{target} not a leaf")
                pass
            elif is_likelihood(target):
                observation_ids = get_observation_ids(target)
                if len(observation_ids) != 1:
                    raise ValueError(f"cannot convert {target} to likelihood")
                observation_id = observation_ids[0]
                observation = observations[observation_id]
                likelihood = channel2likelihood(
                    target, y=observation, y_name=observation_id
                )
                dag.add_edge(source, likelihood)
            else:
                dag.add_edge(source, target)
        return ModelDAG(dag)

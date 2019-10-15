from .callbacks import EarlyStopping
from .initial_conditions import ConstantInit
from ..models import Model
from ..base import Variable, Factor
import numpy as np
import networkx as nx
import logging
logger = logging.getLogger(__name__)


def get_size(x):
    return np.size(x) if len(np.shape(x)) <= 1 else np.shape(x)


def info_arrow(source, target, data, keys):
    if data["direction"] == "fwd":
        m = f"{source}->{target}"
    else:
        m = f"{target}<-{source}"
    if "a" in keys:
        m += f" a={data['a']:.3f}"
    if "n_iter" in keys and "n_iter" in data:
        m += f" n_iter={data['n_iter']}"
    if "b" in keys and "b" in data:
        m += f" b={data['b']}"
    if "b_size" in keys and "b" in data:
        b_size = get_size(data['b'])
        m += f" b_size={b_size}"
    return m


def info_message(message, keys=["a", "n_iter"]):
    if len(message) == 0:
        return "[]"
    infos = [
        info_arrow(source, target, data, keys)
        for source, target, data in message
    ]
    return "\n".join(infos)


def find_variable_in_nodes(id, nodes):
    matchs = [
        node for node in nodes
        if isinstance(node, Variable) and node.id == id
    ]
    assert len(matchs) == 1
    return matchs[0]


class MessagePassing():

    def __init__(self, model, message_keys):
        if not isinstance(model, Model):
            raise ValueError(f"model {model} is not a Model")
        self.message_keys = message_keys
        self.model_dag = model.dag
        self.forward_ordering = model.forward_ordering
        self.backward_ordering = list(reversed(model.forward_ordering))
        self.variables = model.variables
        self.n_iter = 0

    def configure_variables_damping(self, variables_damping):
        """Configure damping options on variables

        Parameters
        ----------
        - variables_damping: list of variable.id, direction, damping tuples
            Factor-to-variable edges into `variable.id` and given `direction`
            will be damped with `damping`.
        """
        for id, direction, damping in variables_damping:
            variable = find_variable_in_nodes(id, self.message_dag.nodes())
            edges = self.message_dag.in_edges(variable, data=True)
            for source, target, data in edges:
                if data["direction"] == direction:
                    data["damping"] = damping
                    logger.info(f"damping {source}->{target} at {damping}")

    def damp_message(self, message):
        "Damp message in-place"
        for source, target, data in message:
            damping = self.message_dag[source][target]["damping"]
            if damping:
                for key in self.message_keys:
                    old_value = self.message_dag[source][target][key]
                    data[key] = damping * old_value + (1 - damping) * data[key]

    def check_message(self, new_message, old_message):
        "Raise error on nan values"
        for source, target, data in new_message:
            if np.isnan(data['a']):
                logger.error(
                    f"{source}->{target} a is nan\n" +
                    "incoming:\n" +
                    info_message(old_message, keys=["n_iter", "a", "b"])
                )
                raise ValueError(f"{source}->{target} a is nan")
            if (data['a'] < 0):
                logger.warning(f"{source}->{target} negative a {data['a']}")
            if ('b' in data) and np.isnan(data['b']).any():
                logger.error(
                    f"{source}->{target} b is nan\n" +
                    "incoming:\n" +
                    info_message(old_message, keys=["n_iter", "a", "b"])
                )
                raise ValueError(f"{source}->{target} b is nan")

    def init_message_dag(self, initializer):
        message_dag = nx.DiGraph()
        message_dag.add_nodes_from(self.model_dag.nodes(data=True))
        message_dag.add_edges_from(
            self.model_dag.edges(data=True), direction="fwd",
            damping=None, n_iter=0
        )
        message_dag.add_edges_from(
            self.model_dag.reverse().edges(data=True), direction="bwd",
            damping=None, n_iter=0
        )
        for source, target, data in message_dag.edges(data=True):
            variable = source if isinstance(source, Variable) else target
            x_data = self.model_dag.node[variable]
            data["tau"] = x_data.get("tau")
            data["shape"] = x_data.get("shape")
            for message_key in self.message_keys:
                data[message_key] = initializer.init(
                    message_key, data["shape"], variable.id, data["direction"]
                )
        self.message_dag = message_dag
        nx.freeze(self.message_dag)

    def reset_message_dag(self, message_dag):
        self.message_dag = message_dag
        self.variables = [
            node for node in message_dag.nodes()
            if isinstance(node, Variable)
        ]

    def update_message(self, new_message):
        for source, target, new_data in new_message:
            n_iter = self.message_dag[source][target]["n_iter"]
            new_data.update(n_iter=n_iter + 1)
            self.message_dag[source][target].update(new_data)

    def forward_message(self):
        for node in self.forward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            new_message = self.forward(node, message)
            self.check_message(new_message, message)
            self.damp_message(new_message)
            self.update_message(new_message)

    def backward_message(self):
        for node in self.backward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            new_message = self.backward(node, message)
            self.check_message(new_message, message)
            self.damp_message(new_message)
            self.update_message(new_message)

    def update_variables(self):
        for variable in self.variables:
            message = self.message_dag.in_edges(variable, data=True)
            new_data = self.update(variable, message)
            self.message_dag.node[variable].update(new_data)

    def get_variables_data(self, ids="all"):
        data = {}
        for variable in self.variables:
            if ids == "all" or variable.id in ids:
                data[variable.id] = self.message_dag.node[variable].copy()
        return data

    def get_variable_data(self, id):
        for variable in self.variables:
            if variable.id == id:
                return self.message_dag.node[variable].copy()
        raise ValueError(f"id={id} not in variables")

    def update_objective(self):
        for node in self.forward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            A = self.node_objective(node, message)
            self.message_dag.node[node].update(A=A)
        for source, target, data in message_dag.edges(data=True):
            if data["direction"] == "fwd":
                variable = source if isinstance(source, Variable) else target
                message = [
                    (source, target, data),
                    (target, source, self.message_dag[target][source])
                ]
                A = self.node_objective(variable, message)
                self.message_dag.node[source][target].update(A=A)
                self.message_dag.node[target][source].update(A=A)
        A_nodes = sum(
            data["A"] for node, data in self.message_dag.nodes(data=True)
        )
        A_edges = sum(
            data["A"] for s, t, data in self.message_dag.edges(data=True)
            if data["direction"] == "fwd"  # avoid double counting !
        )
        self.A_model = A_nodes - A_edges

    def check_any_small(self):
        "Returns True if any v is below epsilon."
        epsilon = 1e-10
        any_small = False
        for variable in self.variables:
            new_v = self.message_dag.node[variable]["v"]
            if new_v < epsilon:
                any_small = True
                logger.info(f"new_v={new_v}<epsilon={epsilon} for {variable}")
        return any_small

    def check_any_nan(self):
        "Returns True if any v is nan."
        any_nan = False
        for variable in self.variables:
            new_v = self.message_dag.node[variable]["v"]
            if np.isnan(new_v):
                any_nan = True
                logger.info(f"new_v is nan for {variable}")
        return any_nan

    def check_any_increasing(self, old_message_dag):
        "Returns True if any v is increasing."
        any_increasing = False
        for variable in self.variables:
            old_variable = find_variable_in_nodes(
                variable.id, old_message_dag.nodes()
            )
            old_v = old_message_dag.node[old_variable]["v"]
            new_v = self.message_dag.node[variable]["v"]
            if new_v > old_v:
                any_increasing = True
                logger.info(f"new_v={new_v}>old_v={old_v} for {variable}")
        return any_increasing

    def iterate(self, max_iter=200,
                callback=None, initializer=None, warm_start=False,
                check_nan=True, check_decreasing=True,
                variables_damping=None):
        initializer = initializer or ConstantInit(a=0, b=0)
        callback = callback or EarlyStopping(tol=1e-6, min_variance=1e-12)
        if warm_start:
            if not hasattr(self, "message_dag"):
                raise ValueError("message dag was not initialized")
            logger.info(f"warm start with n_iter={self.n_iter} - no initialization")
        else:
            logger.info(f"init message dag with {initializer}")
            self.n_iter = 0
            self.init_message_dag(initializer)
        if variables_damping:
            self.configure_variables_damping(variables_damping)
        for i in range(max_iter):
            # backup message_dag
            if (i > 0) and (check_nan or check_decreasing):
                old_message_dag = self.message_dag.copy()
            # forward, backward, update pass
            self.forward_message()
            self.backward_message()
            self.update_variables()
            # early stoppings with restoring message_dag
            if (i > 0) and check_nan:
                any_nan = self.check_any_nan()
                if any_nan:
                    logger.warn("nan v restoring old message dag")
                    self.reset_message_dag(old_message_dag)
                    logger.info(f"terminated after n_iter={self.n_iter} iterations")
                    return
            if (i > 0) and check_decreasing:
                any_increasing = self.check_any_increasing(old_message_dag)
                if any_increasing:
                    logger.warn("increasing v restoring old message dag")
                    self.reset_message_dag(old_message_dag)
                    logger.info(f"terminated after n_iter={self.n_iter} iterations")
                    return
            # callbacks
            self.n_iter += 1
            stop = callback(self, i, max_iter)
            logger.debug(f"n_iter={self.n_iter}")
            if stop:
                logger.info(f"terminated after n_iter={self.n_iter} iterations")
                return

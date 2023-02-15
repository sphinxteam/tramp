from .callbacks import JoinCallback
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
        m = f"{source.id}->{target.id}"
    else:
        m = f"{target.id}<-{source.id}"
    for key in keys:
        if key=="a":
            m += f" a={data['a']:.3f}"
        elif key=="b_size":
            b_size = get_size(data['b'])
            m += f" b_size={b_size}"
        else:
            m += f" {key}={data.get(key)}"
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


def create_message(message, source, target, data):
    new_message = [
        (s, t, data if ((s == source) and (t == target)) else d)
        for s, t, d in message
    ]
    return new_message


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

    def check_message(self, new_message, old_message):
        "Raise error on nan values"
        for source, target, data in new_message:
            if np.isnan(data['a']):
                logger.error(
                    f"{source.id}->{target.id} a is nan\n" +
                    "incoming:\n" +
                    info_message(old_message, keys=["n_iter", "a", "b"])
                )
                logger.warning("restoring old message dag")
                self.reset_message_dag(self.old_message_dag)
                raise ValueError(f"{source.id}->{target.id} a is nan")
            if (data['a'] < 0):
                logger.warning(f"{source.id}->{target.id} negative a {data['a']}")
            if ('b' in data) and np.isnan(data['b']).any():
                logger.error(
                    f"{source.id}->{target.id} b is nan\n" +
                    "incoming:\n" +
                    info_message(old_message, keys=["n_iter", "a", "b"])
                )
                logger.warning("restoring old message dag")
                self.reset_message_dag(self.old_message_dag)
                raise ValueError(f"{source}->{target} b is nan")

    def init_message_dag(self, initializer):
        message_dag = nx.DiGraph()
        message_dag.add_nodes_from(self.model_dag.nodes(data=True))
        message_dag.add_edges_from(
            self.model_dag.edges(data=True), direction="fwd", n_iter=0
        )
        message_dag.add_edges_from(
            self.model_dag.reverse().edges(data=True), direction="bwd", n_iter=0
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
        self.compute_objective()

    def damp_message(self, message):
        "Damp message in-place"
        for source, target, data in message:
            for key in self.message_keys:
                old_value = self.message_dag[source][target][key]
                data[key] = self.damping*old_value + (1 - self.damping)*data[key]    

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

    def get_edges_data(self, keys):
        records = []
        for source, target, data in self.message_dag.edges(data=True):
            variable = source if isinstance(source, Variable) else target
            factor = source if isinstance(source, Factor) else target
            record = dict(x_id=variable.id, f_id=factor.id)
            for key in keys:
                record[key] = data.get(key)
            records.append(record)
        return records

    def get_nodes_data(self, keys):
        records = []
        for node, data in self.message_dag.nodes(data=True):
            node_type = "variable" if isinstance(node, Variable) else "factor"
            record = dict(id=node.id, type=node_type)
            for key in keys:
                record[key] = data.get(key)
            record["n_iter"] = self.n_iter
            records.append(record)
        return records

    def get_variable_data(self, id):
        for variable in self.variables:
            if variable.id == id:
                return self.message_dag.node[variable].copy()
        raise ValueError(f"id={id} not in variables")

    def update_objective(self):
        "Computes A on each node (variable/factor) and edge"
        for node in self.forward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            A = self.node_objective(node, message)
            self.message_dag.node[node].update(A=A)
        for source, target, data in self.message_dag.edges(data=True):
            if data["direction"] == "fwd":
                variable = source if isinstance(source, Variable) else target
                message = [
                    (source, target, data),
                    (target, source, self.message_dag[target][source])
                ]
                A = self.node_objective(variable, message)
                self.message_dag[source][target].update(A=A)
                self.message_dag[target][source].update(A=A)

    def compute_objective(self):
        "Computes A for the model = A_nodes - A_edges"
        A_nodes = sum(
            data["A"] for node, data in self.message_dag.nodes(data=True)
        )
        A_edges = sum(
            data["A"] for s, t, data in self.message_dag.edges(data=True)
            if data["direction"] == "fwd"  # avoid double counting !
        )
        self.A_model = A_nodes - A_edges

    def iterate(self, max_iter=200, callback=None, initializer=None, damping=0):
        # handle arguments
        callback = callback or self.default_stopping
        if isinstance(callback, list):
            callback = JoinCallback(callback)
        initializer = initializer or ConstantInit(a=0, b=0)
        logger.info(f"init message dag with {initializer}")
        self.init_message_dag(initializer)
        self.n_iter = 0
        self.damping = damping
        # main loop
        for i in range(max_iter):
            # forward, backward, update pass
            self.forward_message()
            self.backward_message()
            self.update_variables()
            self.update_objective()
            self.compute_objective()
            # callbacks
            self.n_iter += 1
            stop = callback(self, i)
            if stop:
                logger.info(f"terminated after n_iter={self.n_iter} iterations")
                return
            self.old_message_dag = self.message_dag.copy()
        logger.info(f"terminated after n_iter={self.n_iter} iterations")

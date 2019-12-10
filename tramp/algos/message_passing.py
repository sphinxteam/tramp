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

    def configure_damping(self, damping):
        """Configure damping options

        Parameters
        ----------
        - damping: str 'adaptive', float or list
            - str 'adaptive': adaptive damping
            - float: global damping
            - list of variable.id, direction, damping tuples
            Factor-to-variable edges into `variable.id` and given `direction`
            will be damped with `damping`.
        """
        if not damping:
            self.damping = False
            logger.info("no damping")
            return
        self.damping = True
        self.adaptive_damping = (damping == "adaptive")
        if self.adaptive_damping:
            logger.info("adaptive damping activated")
            return
        if not (isinstance(damping, float) or isinstance(damping, list)):
            raise ValueError("damping must be 'adaptive', float or list")
        if isinstance(damping, float):
            x_ids = [variable.id for variable in self.variables]
            damping = [
                (x_id, "fwd", damping) for x_id in x_ids
            ] + [
                (x_id, "bwd", damping) for x_id in x_ids
            ]
        for id, direction, damp in damping:
            variable = find_variable_in_nodes(id, self.message_dag.nodes())
            edges = self.message_dag.in_edges(variable, data=True)
            for source, target, data in edges:
                if data["direction"] == direction:
                    data["damping"] = damp
                    logger.info(info_arrow(source, target, data, ["damping"]))

    def damp_message(self, message):
        "Damp message in-place"
        if not self.damping:
            return
        for source, target, data in message:
            if self.adaptive_damping:
                data_new = self.compute_adaptive_damping(source, target, data)
            else:
                data_new = self.compute_constant_damping(source, target, data)
            data.update(data_new)

    def compute_constant_damping(self, source, target, data):
        damping = self.message_dag[source][target]["damping"]
        if not damping:
            return data
        data_new = data.copy()
        for key in self.message_keys:
            old_value = self.message_dag[source][target][key]
            data_new[key] = damping*old_value + (1 - damping)*data[key]
        return data_new

    def compute_dA(self, source, target, data):
        if self.n_iter == 0:
            return 0
        variable = target if isinstance(target, Variable) else source
        data_old = self.message_dag[source][target]
        m_target_old = self.message_dag.in_edges(target, data=True)
        m_edge_old = [
            (source, target, self.message_dag[source][target]),
            (target, source, self.message_dag[target][source])
        ]
        A_old_target = self.node_objective(target, m_target_old)
        A_old_edge = self.node_objective(variable, m_edge_old)
        A_old = A_old_target - A_old_edge
        data_new = data
        m_target_new = create_message(m_target_old, source, target, data_new)
        m_edge_new = create_message(m_edge_old, source, target, data_new)
        A_new_target = self.node_objective(target, m_target_new)
        A_new_edge = self.node_objective(variable, m_edge_new)
        A_new = A_new_target - A_new_edge
        dA = A_new - A_old
        return dA

    def compute_adaptive_damping(self, source, target, data):
        if self.n_iter == 0:
            return data
        variable = target if isinstance(target, Variable) else source
        data_old = self.message_dag[source][target]
        data_var = {
            key: data[key] - data_old[key] for key in self.message_keys
        }
        m_target_old = self.message_dag.in_edges(target, data=True)
        m_edge_old = [
            (source, target, self.message_dag[source][target]),
            (target, source, self.message_dag[target][source])
        ]
        A_old_target = self.node_objective(target, m_target_old)
        A_old_edge = self.node_objective(variable, m_edge_old)
        A_old = A_old_target - A_old_edge
        data_new = data.copy()
        n_max = 10
        for n in range(n_max):
            beta = 1 / 2**n
            for key in self.message_keys:
                data_new[key] = data_old[key] + beta*data_var[key]
            m_target_new = create_message(m_target_old, source, target, data_new)
            m_edge_new = create_message(m_edge_old, source, target, data_new)
            A_new_target = self.node_objective(target, m_target_new)
            A_new_edge = self.node_objective(variable, m_edge_new)
            A_new = A_new_target - A_new_edge
            dA = A_new - A_old
            if dA >= 0:
                data_new.update(dA=dA, beta=beta)
                return data_new
        # if damping wasn't enough, return old value
        data_new = data_old.copy()
        data_new.update(dA=0, beta=0)
        return data_new

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
            if self.update_dA:
                new_data["dA"] = self.compute_dA(source, target, new_data)
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
        A_nodes = sum(
            data["A"] for node, data in self.message_dag.nodes(data=True)
        )
        A_edges = sum(
            data["A"] for s, t, data in self.message_dag.edges(data=True)
            if data["direction"] == "fwd"  # avoid double counting !
        )
        self.A_model = A_nodes - A_edges

    def iterate(self, max_iter=200,
                callback=None, initializer=None, damping=None,
                warm_start=False, update_dA=False):
        initializer = initializer or ConstantInit(a=0, b=0)
        callback = callback or self.default_stopping
        if warm_start:
            if not hasattr(self, "message_dag"):
                raise ValueError("message dag was never initialized")
            logger.info(f"warm start with n_iter={self.n_iter} no initialization")
        else:
            logger.info(f"init message dag with {initializer}")
            self.init_message_dag(initializer)
            self.n_iter = 0
        self.configure_damping(damping)
        self.update_dA = update_dA
        for i in range(max_iter):
            # forward, backward, update pass
            self.forward_message()
            self.backward_message()
            self.update_variables()
            # callbacks
            self.n_iter += 1
            stop = callback(self, i, max_iter)
            if stop:
                logger.info(f"terminated after n_iter={self.n_iter} iterations")
                return
            self.old_message_dag = self.message_dag.copy()
        logger.info(f"terminated after n_iter={self.n_iter} iterations")

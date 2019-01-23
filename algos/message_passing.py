import numpy as np
import networkx as nx
import logging
from ..base import Variable
from ..models import DAGModel
from .initial_conditions import ConstantInit
from .callbacks import PassCallback


class MessagePassing():
    _default_initializer = ConstantInit()
    _default_callback = PassCallback()

    def __init__(self, model, message_keys, forward, backward, update):
        if not isinstance(model, DAGModel):
            raise ValueError(f"model {model} is not a DAGModel")
        model.second_moment()
        self.message_keys = message_keys
        self.forward = forward
        self.backward = backward
        self.update = update
        self.model_dag = model.model_dag
        self.forward_ordering = model.forward_ordering
        self.backward_ordering = model.backward_ordering
        self.variables = model.variables
        self.n_iter = 0

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
            if data["direction"] == "fwd" and isinstance(source, Variable):
                data["tau"] = self.model_dag.node[source]["tau"]
            variable = source if isinstance(source, Variable) else target
            for message_key in self.message_keys:
                data[message_key] = initializer.init(message_key, variable.shape)
        self.message_dag = message_dag
        nx.freeze(self.message_dag)

    def update_message(self, new_message):
        for source, target, new_data in new_message:
            n_iter = self.message_dag[source][target]["n_iter"]
            new_data.update(n_iter=n_iter + 1)
            self.message_dag[source][target].update(new_data)

    def forward_message(self):
        for node in self.forward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            new_message = self.forward(node, message)
            self.update_message(new_message)

    def backward_message(self):
        for node in self.backward_ordering:
            message = self.message_dag.in_edges(node, data=True)
            new_message = self.backward(node, message)
            self.update_message(new_message)

    def update_variables(self):
        for variable in self.variables:
            message = self.message_dag.in_edges(variable, data=True)
            new_data = self.update(variable, message)
            self.message_dag.node[variable].update(new_data)

    def get_variables_data(self):
        variables_data = []
        for variable in self.variables:
            data = dict(id=variable.id, variable=variable)
            data.update(self.message_dag.node[variable])
            variables_data.append(data)
        return variables_data

    def iterate(self, max_iter, callback=None, initializer=None, warm_start=False):
        initializer = initializer or self._default_initializer
        callback = callback or self._default_callback
        if warm_start:
            if not hasattr(self, "message_dag"):
                raise ValueError("message dag was not initialized")
            logging.info(f"warm start with n_iter={self.n_iter} - no initialization")
        else:
            logging.info(f"init message dag with {initializer}")
            self.n_iter = 0
            self.init_message_dag(initializer)
        for i in range(max_iter):
            self.forward_message()
            self.backward_message()
            self.update_variables()
            self.n_iter += 1
            stop = callback(self, i, max_iter)
            logging.debug(f"n_iter={self.n_iter}")
            if stop:
                logging.info(f"terminated after n_iter={self.n_iter} iterations")
                return

from .initial_conditions import ConstantInit
from .message_passing import MessagePassing, info_message
import numpy as np
import logging
logger = logging.getLogger(__name__)


class ExplainMessagePassing(MessagePassing):
    def __init__(self, model, keys=[],
                 print_incoming=True, print_outcoming=True):
        model.init_shapes()
        super().__init__(model, message_keys=["a", "b"])
        self.keys = keys
        self.print_incoming = print_incoming
        self.print_outcoming = print_outcoming

    def forward(self, node, message):
        if self.print_incoming:
            print(f"{node}: incoming message")
            print(info_message(message, self.keys))
        new_message = node.forward_message(message)
        if self.print_outcoming:
            print(f"{node}: outgoing message")
            print(info_message(new_message, self.keys))
        return new_message

    def backward(self, node, message):
        if self.print_incoming:
            print(f"{node}: incoming message")
            print(info_message(message, self.keys))
        new_message = node.backward_message(message)
        if self.print_outcoming:
            print(f"{node}: outgoing message")
            print(info_message(new_message, self.keys))
        return new_message

    def update(self, variable, message):
        r, v = variable.posterior_rv(message)
        return dict(r=r, v=v)

    def run(self, n_iter=1, initializer=None):
        initializer = initializer or ConstantInit(a=0, b=0)
        logger.info(f"init message dag with {initializer}")
        self.init_message_dag(initializer)
        for _ in range(n_iter):
            print("FORWARD PASS")
            print("-" * len("FORWARD PASS"))
            self.forward_message()
            print("BACKWARD PASS")
            print("-" * len("BACKWARD PASS"))
            self.backward_message()

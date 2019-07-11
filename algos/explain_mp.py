import numpy as np
import logging
from .message_passing import MessagePassing, info_message
from .initial_conditions import ConstantInit


class ExplainMessagePassing(MessagePassing):
    def __init__(self, model, keys=[],
                 print_incoming=True, print_outcoming=True):
        def forward(node, message):
            if print_incoming:
                print(f"{node}: incoming message")
                print(info_message(message, keys))
            new_message = node.forward_message(message)
            if print_outcoming:
                print(f"{node}: outgoing message")
                print(info_message(new_message, keys))
            return new_message

        def backward(node, message):
            if print_incoming:
                print(f"{node}: incoming message")
                print(info_message(message, keys))
            new_message = node.backward_message(message)
            if print_outcoming:
                print(f"{node}: outgoing message")
                print(info_message(new_message, keys))
            return new_message

        def update(variable, message):
            r, v = variable.posterior_rv(message)
            return dict(r=r, v=v)

        super().__init__(
            model, message_keys=["a", "b"],
            forward=forward, backward=backward, update=update
        )

    def run(self, n_iter=1, initializer=None):
        initializer = initializer or ConstantInit(a=0, b=0)
        logging.info(f"init message dag with {initializer}")
        self.init_message_dag(initializer)
        for _ in range(n_iter):
            print("FORWARD PASS")
            print("-" * len("FORWARD PASS"))
            self.forward_message()
            print("BACKWARD PASS")
            print("-" * len("BACKWARD PASS"))
            self.backward_message()

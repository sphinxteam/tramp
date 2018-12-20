import numpy as np
import logging
from .message_passing import MessagePassing


def get_size(x):
    return np.size(x) if len(np.shape(x)) <= 1 else np.shape(x)


def info_message(message, keys=["a", "n_iter"]):

    if len(message) == 0:
        return "[]"

    def info(source, target, data):
        m = f"{source}->{target}" if data["direction"] == "fwd" else f"{target}<-{source}"
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

    infos = [info(source, target, data) for source, target, data in message]
    return "\n".join(infos)


class ExplainMessagePassing(MessagePassing):
    def __init__(self, model, keys=[]):
        def forward(node, message):
            print(f"{node}: incoming message")
            print(info_message(message, keys))
            new_message = node.forward_message(message)
            print(f"{node}: outgoing message")
            print(info_message(new_message, keys))
            return new_message

        def backward(node, message):
            print(f"{node}: incoming message")
            print(info_message(message, keys))
            new_message = node.backward_message(message)
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

    def run(self):

        initializer = self._default_initializer
        logging.info(f"init message dag with {initializer}")
        self.init_message_dag(initializer)

        print("FORWARD PASS")
        print("-" * len("FORWARD PASS"))
        self.forward_message()
        print("BACKWARD PASS")
        print("-" * len("BACKWARD PASS"))
        self.backward_message()

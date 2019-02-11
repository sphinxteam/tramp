from .message_passing import MessagePassing


class StateEvolution(MessagePassing):
    def __init__(self, model):
        def forward(node, message):
            return node.forward_state_evolution(message)

        def backward(node, message):
            return node.backward_state_evolution(message)

        def update(variable, message):
            v = variable.posterior_v(message)
            return dict(v=v)
        super().__init__(
            model, message_keys=["a"],
            forward=forward, backward=backward, update=update
        )

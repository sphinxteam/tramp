from .message_passing import MessagePassing


class ExpectationPropagation(MessagePassing):
    def __init__(self, model):
        def forward(node, message):
            return node.forward_message(message)

        def backward(node, message):
            return node.backward_message(message)

        def update(variable, message):
            r, v = variable.posterior_rv(message)
            return dict(r=r, v=v)
        super().__init__(
            model, message_keys=["a", "b"],
            forward=forward, backward=backward, update=update
        )

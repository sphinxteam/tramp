from .message_passing import MessagePassing


class ExpectationPropagation(MessagePassing):
    def __init__(self, model):
        model.init_shapes()
        super().__init__(model, message_keys=["a", "b"])

    def forward(self, node, message):
        return node.forward_message(message)

    def backward(self, node, message):
        return node.backward_message(message)

    def update(self, variable, message):
        r, v = variable.posterior_rv(message)
        return dict(r=r, v=v)

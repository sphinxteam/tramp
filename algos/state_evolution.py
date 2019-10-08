from .message_passing import MessagePassing


class StateEvolution(MessagePassing):
    def __init__(self, model):
        model.init_second_moments()
        super().__init__(model, message_keys=["a"])

    def forward(self, node, message):
        return node.forward_state_evolution(message)

    def backward(self, node, message):
        return node.backward_state_evolution(message)

    def update(self, variable, message):
        v = variable.posterior_v(message)
        return dict(v=v)

    def node_objective(self, node, message):
        return node.free_energy(message)

    def entropy(self, update=True):
        if update:
            self.update_objective()
        return -self.A_model

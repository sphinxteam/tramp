from .message_passing import MessagePassing
from .callbacks import EarlyStopping, EarlyStoppingEP, JoinCallback


class ExpectationPropagation(MessagePassing):
    def __init__(self, model):
        model.init_shapes()
        super().__init__(model, message_keys=["a", "b"])
        self.default_stopping = EarlyStoppingEP()

    def forward(self, node, message):
        return node.forward_message(message)

    def backward(self, node, message):
        return node.backward_message(message)

    def update(self, variable, message):
        r, v = variable.posterior_rv(message)
        return dict(r=r, v=v)

    def node_objective(self, node, message):
        return node.log_partition(message)

    def log_evidence(self, update=True):
        if update:
            self.update_objective()
        return self.A_model

    def surprisal(self, update=True):
        if update:
            self.update_objective()
        return -self.A_model

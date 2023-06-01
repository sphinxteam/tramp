from .message_passing import MessagePassing
from .callbacks import EarlyStopping, EarlyStoppingEP, JoinCallback
from ..base import Variable, Factor

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

    def log_evidence(self):
        return self.A_model

    def surprisal(self):
        return -self.A_model
    
    def compute_m_condition(self):
        "Computes m-condition (on mean r)"
        for source, target, data in self.message_dag.edges(data=True):
            if data["direction"] == "fwd":
                variable = source if isinstance(source, Variable) else target
                factor = source if isinstance(source, Factor) else target
                edge_message = [
                    (source, target, data),
                    (target, source, self.message_dag[target][source])
                ]
                rx, vx = variable.posterior_rv(edge_message)
                factor_message = self.message_dag.in_edges(factor, data=True)
                moments = factor.compute_moments(factor_message)
                x_moment = [(s, r, v) for s, r, v in moments if s==variable]
                assert len(x_moment)==1
                _, rfx, vfx = x_moment[0]
                m = ((rfx - rx)**2).mean() 
                self.message_dag[source][target].update(m=m)
        return sum(
            data["m"] for s, t, data in self.message_dag.edges(data=True)
            if data["direction"] == "fwd"  # avoid double counting !
        )
from .message_passing import MessagePassing
from .initial_conditions import ConstantInit


def format_latex_message(message, comment):
    bwd_sources = [
        source for source, target, data in message
        if data["direction"] == "fwd"
    ]
    fwd_sources = [
        source for source, target, data in message
        if data["direction"] == "bwd"
    ]
    targets = [
        target for source, target, data in message
    ]
    m = r"\mathrm{" + comment + r"}\;"
    m += ",".join(source.math()[1:-1] for source in set(bwd_sources))
    if bwd_sources:
        m += r" \rightarrow "
    m += ",".join(target.math()[1:-1] for target in set(targets))
    if fwd_sources:
        m += r" \leftarrow "
    m += ",".join(source.math()[1:-1] for source in set(fwd_sources))
    return m


class DisplayLatexMessagePassing(MessagePassing):
    def __init__(self, model):
        model.init_shapes()
        super().__init__(model, message_keys=["a", "b"])

    def forward(self, node, message):
        m = format_latex_message(message, "incoming")
        new_message = node.forward_message(message)
        m += r"\;" + format_latex_message(new_message, "outcoming")
        self.latex["forward"].append(r"$" + m + r"$")
        return new_message

    def backward(self, node, message):
        m = format_latex_message(message, "incoming")
        new_message = node.backward_message(message)
        m += r"\;" + format_latex_message(new_message, "outcoming")
        self.latex["backward"].append(r"$" + m + r"$")
        return new_message

    def update(self, variable, message):
        pass

    def run(self):
        self.latex = dict(forward=[], backward=[])
        initializer = ConstantInit(a=0, b=0)
        self.init_message_dag(initializer)
        self.configure_damping(None)
        self.update_dA = False
        self.forward_message()
        self.backward_message()
        return self.latex

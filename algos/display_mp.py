from .message_passing import MessagePassing


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
    for source in set(bwd_sources):
        m += source.math()[1:-1]
    if bwd_sources:
        m += r" \rightarrow "
    for target in set(targets):
        m += target.math()[1:-1]
    if fwd_sources:
        m += r" \leftarrow "
    for source in set(fwd_sources):
        m += source.math()[1:-1]
    return m


class DisplayLatexMessagePassing(MessagePassing):
    def __init__(self, model, keys=[]):
        def forward(node, message):
            m = format_latex_message(message, "incoming")
            new_message = node.forward_message(message)
            m += r"\;" + format_latex_message(new_message, "outcoming")
            self.latex["forward"].append(r"$" + m + r"$")
            return new_message

        def backward(node, message):
            m = format_latex_message(message, "incoming")
            new_message = node.backward_message(message)
            m += r"\;" + format_latex_message(new_message, "outcoming")
            self.latex["backward"].append(r"$" + m + r"$")
            return new_message

        def update(variable, message):
            pass

        super().__init__(
            model, message_keys=["a", "b"],
            forward=forward, backward=backward, update=update
        )

    def run(self):
        self.latex = dict(forward=[], backward=[])
        initializer = self._default_initializer
        self.init_message_dag(initializer)
        self.forward_message()
        self.backward_message()
        return self.latex

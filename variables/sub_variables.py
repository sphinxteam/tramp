from ..base import Variable


class SIMOVariable(Variable):

    def __init__(self, n_next, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=n_next, dtype=dtype, id=id)


class MISOVariable(Variable):

    def __init__(self, n_prev, dtype=float, id=None):
        super().__init__(n_prev=n_prev, n_next=1, dtype=dtype, id=id)


class SISOVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=1, dtype=dtype, id=id)

    def forward_message(self, message):
        "pass message from previous factor k to next factor l"
        k_source, l_source, ak, bk, al, bl = self._parse_message_ab(message)
        new_message = [(self, l_source, dict(a=ak, b=bk, direction="fwd"))]
        return new_message

    def backward_message(self, message):
        "pass message from next factor l to previous factor k"
        k_source, l_source, ak, bk, al, bl = self._parse_message_ab(message)
        new_message = [(self, k_source, dict(a=al, b=bl, direction="bwd"))]
        return new_message

    def forward_state_evolution(self, message):
        "pass message from previous factor k to next factor l"
        k_source, l_source, ak, al = self._parse_message_a(message)
        new_message = [(self, l_source, dict(a=ak, direction="fwd"))]
        return new_message

    def backward_state_evolution(self, message):
        "pass message from next factor l to previous factor k"
        k_source, l_source, ak, al = self._parse_message_a(message)
        new_message = [(self, k_source, dict(a=al, direction="bwd"))]
        return new_message


class MILeafVariable(Variable):

    def __init__(self, n_prev, dtype=float, id=None):
        super().__init__(n_prev=n_prev, n_next=0, dtype=dtype, id=id)


class SILeafVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=0, dtype=dtype, id=id)


class MORootVariable(Variable):

    def __init__(self, n_next, dtype=float, id=None):
        super().__init__(n_prev=0, n_next=n_next, dtype=dtype, id=id)


class SORootVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=0, n_next=1, dtype=dtype, id=id)

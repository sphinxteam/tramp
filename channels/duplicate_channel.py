from ..base import Factor


class DuplicateChannel(Factor):
    n_next = 2
    n_prev = 1

    def __init__(self):
        self.repr_init()

    def sample(self, Z):
        return Z, Z

    def math(self):
        return r"$\delta$"

    def second_moment(self, tau):
        return tau, tau

    def forward_posterior(self, message):
        # estimate x, y from x = y = z
        a = sum(data["a"] for source, target, data in message)
        b = sum(data["b"] for source, target, data in message)
        r = b / a
        v = 1. / a
        return [(r, v), (r, v)]

    def backward_posterior(self, message):
        # estimate z from x = y = z
        a = sum(data["a"] for source, target, data in message)
        b = sum(data["b"] for source, target, data in message)
        r = b / a
        v = 1. / a
        return [(r, v)]

    def proba_beliefs(self, message):
        raise NotImplementedError

import numpy as np
from ..base import Channel


class GaussianChannel(Channel):
    def __init__(self, var=1):
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var

    def sample(self, Z):
        noise = self.sigma * np.random.standard_normal(Z.shape)
        X = Z + noise
        return X

    def math(self):
        return r"$\mathcal{N}$"

    def second_moment(self, tau):
        return tau + self.var

    def forward_message(self, message):
        # x = z + noise, x source variable of bwd message
        az, bz, ax, bx = self._parse_message_ab(message)
        source, target = self._parse_endpoints(message, "bwd")
        kz = self.a / (self.a + az)
        new_data = dict(a=kz * az, b=kz * bz, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_message(self, message):
        # z = x - noise, z source variable of fwd message
        az, bz, ax, bx = self._parse_message_ab(message)
        source, target = self._parse_endpoints(message, "fwd")
        kx = self.a / (self.a + ax)
        new_data = dict(a=kx * ax, b=kx * bx, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

    def forward_state_evolution(self, message):
        # x = z + noise, x source variable of bwd message
        az, ax = self._parse_message_a(message)
        source, target = self._parse_endpoints(message, "bwd")
        kz = self.a / (self.a + az)
        new_data = dict(a=kz * az, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_state_evolution(self, message):
        # z = x - noise, z source variable of fwd message
        az, ax = self._parse_message_a(message)
        source, target = self._parse_endpoints(message, "fwd")
        kx = self.a / (self.a + ax)
        new_data = dict(a=kx * ax, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

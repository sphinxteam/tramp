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
        source, target = self._parse_endpoints(message, "bwd")
        zdata, xdata = self._parse_message(message)
        az, bz = zdata["a"], zdata["b"]
        kz = self.a / (self.a + az)
        new_data = dict(a=kz * az, b=kz * bz, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_message(self, message):
        # z = x - noise, z source variable of fwd message
        source, target = self._parse_endpoints(message, "fwd")
        zdata, xdata = self._parse_message(message)
        ax, bx = xdata["a"], xdata["b"]
        kx = self.a / (self.a + ax)
        new_data = dict(a=kx * ax, b=kx * bx, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

    def forward_state_evolution(self, message):
        # x = z + noise, x source variable of bwd message
        source, target = self._parse_endpoints(message, "bwd")
        zdata, xdata = self._parse_message(message)
        az = zdata["a"]
        kz = self.a / (self.a + az)
        new_data = dict(a=kz * az, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_state_evolution(self, message):
        # z = x - noise, z source variable of fwd message
        source, target = self._parse_endpoints(message, "fwd")
        zdata, xdata = self._parse_message(message)
        ax = xdata["a"]
        kx = self.a / (self.a + ax)
        new_data = dict(a=kx * ax, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

import numpy as np
from ..base import Prior
from ..utils.integration import gaussian_measure

class GaussianPrior(Prior):
    def __init__(self, size, mean=0, var=1):
        self.size = size
        self.mean = mean
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = mean / var * np.ones(size)

    def sample(self):
        X = self.mean + self.sigma * np.random.standard_normal(self.size)
        return X

    def math(self):
        return r"$\mathcal{N}$"

    def second_moment(self):
        return self.mean**2 + self.var

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        rx = b / a
        vx = 1 / a
        return rx, vx

    def forward_message(self, message):
        source, target = self._parse_endpoints(message)
        new_data = dict(a=self.a, b=self.b, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def forward_state_evolution(self, message):
        source, target = self._parse_endpoints(message)
        new_data = dict(a=self.a, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def measure(self, f):
        return gaussian_measure(self.mean, self.sigma, f)

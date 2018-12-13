import numpy as np
from ..base import Likelihood


class GaussianLikelihood(Likelihood):
    def __init__(self, y, var=1):
        self.size = y.shape[0] if len(y.shape) == 1 else y.shape
        self.var = var
        self.repr_init()
        self.y = y
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = y / var

    def sample(self, X):
        noise = self.sigma * np.random.standard_normal(X.shape)
        return X + noise

    def math(self):
        return r"$\mathcal{N}$"

    def backward_message(self, message):
        source, target = self._parse_endpoints(message)
        new_data = dict(a=self.a, b=self.b, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_state_evolution(self, message):
        source, target = self._parse_endpoints(message)
        new_data = dict(a=self.a, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

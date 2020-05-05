import numpy as np
from .base_prior import Prior


class ExponentialPrior(Prior):
    """
        - lambda * exp(-lambda * x)
        - mean = 1 / lambda 
        - var = 1 / lambda^2
    """

    def __init__(self, size, mean=0):
        self.size = size
        self.mean = mean
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 0
        self.b = mean

    def sample(self):
        X = np.random.exponential(scale=1/mean, self.size)
        return X

    def math(self):
        return r"\exp"

    def second_moment(self):
        return self.mean**2

    def compute_forward_posterior(self, ax, bx):
        a = ax + self.a
        b = bx + self.b
        rx = - 1 / b
        vx = 1 / b**2
        return rx, vx

    def compute_forward_error(self, ax):
        # a = ax + self.a
        # vx = 1 / a
        raise NotImplementedError
        return vx

    def compute_forward_message(self, ax, bx):
        # ax_new = self.a
        # bx_new = self.b * np.ones_like(bx)
        raise NotImplementedError
        return ax_new, bx_new

    def compute_forward_state_evolution(self, ax):
        # ax_new = self.a
        raise NotImplementedError
        return ax_new

    def measure(self, f):
        raise NotImplementedError

    def compute_log_partition(self, ax, bx):
        # a = ax + self.a
        # b = bx + self.b
        # logZ = 0.5 * np.sum(
        #     b**2 / a - self.b**2 / self.a + np.log(self.a/a)
        # )
        raise NotImplementedError
        return logZ

    def compute_mutual_information(self, ax):
        # a = ax + self.a
        # I = 0.5*np.log(a*self.var)
        raise NotImplementedError
        return I

    def compute_free_energy(self, ax):
        # tau_x = self.second_moment()
        # I = self.compute_mutual_information(ax)
        # A = 0.5*ax*tau_x - I
        raise NotImplementedError
        return A

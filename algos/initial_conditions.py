from ..base import ReprMixin, Variable, Factor
import numpy as np


class InitialConditions(ReprMixin):
    def init(self, message_key, shape):
        if message_key == "a":
            return self.init_a(shape)
        if message_key == "b":
            return self.init_b(shape)


class ConstantInit(InitialConditions):
    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b
        self.repr_init()

    def init_a(self, shape):
        return self.a

    def init_b(self, shape):
        assert shape is not None
        return self.b * np.ones(shape)


class NoisyInit(InitialConditions):
    def __init__(self, a_mean=0, a_var=0, b_mean=0, b_var=1):
        self.a_mean = a_mean
        self.a_var = a_var
        self.b_mean = b_mean
        self.b_var = b_var
        self.repr_init()
        self.a_sigma = np.sqrt(a_var)
        self.b_sigma = np.sqrt(b_var)

    def init_a(self, shape):
        return self.a_mean + self.a_sigma * np.random.standard_normal()

    def init_b(self, shape):
        assert shape is not None
        return self.b_mean + self.b_sigma * np.random.standard_normal(shape)

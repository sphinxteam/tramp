from ..base import ReprMixin, Variable, Factor
import numpy as np


class InitialConditions(ReprMixin):
    def init(self, message_key, shape, id, direction):
        if message_key == "a":
            return self.init_a(shape, id, direction)
        if message_key == "b":
            return self.init_b(shape, id, direction)


class ConstantInit(InitialConditions):
    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b
        self.repr_init()

    def init_a(self, shape, id, direction):
        return self.a

    def init_b(self, shape, id, direction):
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

    def init_a(self, shape, id, direction):
        return self.a_mean + self.a_sigma * np.random.standard_normal()

    def init_b(self, shape, id, direction):
        assert shape is not None
        return self.b_mean + self.b_sigma * np.random.standard_normal(shape)


class CustomInit(InitialConditions):
    """Custom init on variables

    Parameters
    ----------
    - a_init: list of variable.id, direction, a tuples
        Edges from/into `variable.id` and given `direction`
        will be initialized with a = `a`
    - b_init: list of variable.id, direction, b tuples
        Edges from/into `variable.id` and given `direction`
        will be initialized with b = `b`
    - a : float
        Default constant value for a.
    - b : float
        Default constant value for b.
    """

    def __init__(self, a_init=None, b_init=None, a=0, b=0):
        a_init = a_init or []
        self.a_init = {id: {direction: a} for id, direction, a in a_init}
        b_init = b_init or []
        self.b_init = {id: {direction: b} for id, direction, b in b_init}
        self.a = a
        self.b = b
        self.repr_init()

    def init_a(self, shape, id, direction):
        try:
            a = self.a_init[id][direction]
        except KeyError:
            a = self.a
        return a

    def init_b(self, shape, id, direction):
        assert shape is not None
        try:
            b = self.b_init[id][direction]
            assert b.shape == shape
        except KeyError:
            b = self.b * np.ones(shape)
        return b

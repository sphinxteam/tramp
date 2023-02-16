from ..base import ReprMixin, Variable, Factor
import numpy as np


class InitialConditions(ReprMixin):
    def init(self, message_key, shape, source_id, target_id):
        if message_key == "a":
            return self.init_a(shape, source_id, target_id)
        if message_key == "b":
            return self.init_b(shape, source_id, target_id)


class ConstantInit(InitialConditions):
    def __init__(self, a=0, b=0):
        self.a = a
        self.b = b
        self.repr_init()

    def init_a(self, shape, source_id, target_id):
        return self.a

    def init_b(self, shape, source_id, target_id):
        assert shape is not None
        return self.b * np.ones(shape)


class NoisyInit(InitialConditions):
    def __init__(self, a_mean=0, a_std=0, b_mean=0, b_std=1):
        self.a_mean = a_mean
        self.a_std = a_std
        self.b_mean = b_mean
        self.b_std = b_std
        self.repr_init()

    def init_a(self, shape, source_id, target_id):
        return self.a_mean + self.a_std * np.random.standard_normal()

    def init_b(self, shape, source_id, target_id):
        assert shape is not None
        return self.b_mean + self.b_std * np.random.standard_normal(shape)


class CustomInit(InitialConditions):
    """Custom init on variables

    Parameters
    ----------
    - a_init: dict[str, float]
        Keys represent edges, for example {"x->f":a1, "g->z":a2} will initialize 
        the message "x->f" with a=a1 and the message "g->z" with a=a2.
    - b_init: dict[str, array]
        Keys represent edges, for example {"x->f":b1, "g->z":b2} will initialize 
        the message "x->f" with b=b1 and the message "g->z" with b=b2.
    - a : float
        Default constant value for a.
    - b : float
        Default constant value for b.
    """

    def __init__(self, a_init={}, b_init={}, a=0, b=0):
        self.a_init = a_init
        self.b_init = b_init
        self.a = a
        self.b = b
        self.repr_init()

    def init_a(self, shape, source_id, target_id):
        edge = f"{source_id}->{target_id}"
        return self.a_init.get(edge, self.a)

    def init_b(self, shape, source_id, target_id):
        assert shape is not None
        edge = f"{source_id}->{target_id}"
        b = self.b_init.get(edge, self.b*np.ones(shape))
        assert b.shape == shape
        return b

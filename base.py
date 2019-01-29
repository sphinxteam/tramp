"""
Base classes.
"""

import numpy as np
from scipy.integrate import quad, dblquad
import logging


class ReprMixin():
    def repr_init(self, pad=None):
        self._repr_kwargs = self.__dict__.copy()
        self._repr_pad = pad

    def __repr__(self):
        if self._repr_pad:
            pad = f"\n{self._repr_pad}"
        else:
            pad = ""
        sep = ","
        args = sep.join(
            f"{pad}{key}={val}" for key, val in self._repr_kwargs.items()
        )
        if self._repr_pad:
            args += "\n"
        name = self.__class__.__name__
        return f"{name}({args})"


# NOTE : message = [source,target,data]
def filter_message(message, direction):
    filtered_message = [
        (source, target, data)
        for source, target, data in message
        if data["direction"] == direction
    ]
    return filtered_message


class Variable(ReprMixin):
    def __init__(self, shape, dtype=float, id=None):
        self.id = id
        self.size = shape[0] if len(shape) == 1 else shape
        self.repr_init()
        self.shape = shape
        self.dtype = dtype

    def math(self):
        str_id = "?" if self.id is None else str(self.id)
        return r"$X_{" + str_id + r"}$"

    def check_message(self, message):
        for source, target, data in message:
            if (target != self):
                raise ValueError(f"target {target} is not the instance {self}")
            if not isinstance(source, Factor):
                raise ValueError(f"source {source} is not a Factor")
        n_next = len(filter_message(message, "bwd"))
        n_prev = len(filter_message(message, "fwd"))
        if (self.n_next != n_next):
            raise ValueError(
                f"number of next factors : expected {self.n_next} got {n_next}")
        if (self.n_prev != n_prev):
            raise ValueError(
                f"number of prev factors : expected {self.n_prev} got {n_prev}")

    def posterior_ab(self, message):
        a_hat = sum(data["a"] for source, target, data in message)
        b_hat = sum(data["b"] for source, target, data in message)
        return a_hat, b_hat

    def posterior_rv(self, message):
        a_hat, b_hat = self.posterior_ab(message)
        r_hat = b_hat / a_hat
        v_hat = 1. / a_hat
        return r_hat, v_hat

    def posterior_a(self, message):
        a_hat = sum(data["a"] for source, target, data in message)
        return a_hat

    def posterior_v(self, message):
        a_hat = self.posterior_a(message)
        v_hat = 1. / a_hat
        return v_hat

    def forward_message(self, message):
        if self.n_next == 0:
            return []
        a_hat, b_hat = self.posterior_ab(message)
        bwd_message = filter_message(message, "bwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], b=b_hat - data["b"], direction="fwd"))
            for source, target, data in bwd_message
        ]
        return new_message

    def backward_message(self, message):
        if self.n_prev == 0:
            return []
        a_hat, b_hat = self.posterior_ab(message)
        fwd_message = filter_message(message, "fwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], b=b_hat - data["b"], direction="bwd"))
            for source, target, data in fwd_message
        ]
        return new_message

    def forward_state_evolution(self, message):
        if self.n_next == 0:
            return []
        a_hat = self.posterior_a(message)
        bwd_message = filter_message(message, "bwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], direction="fwd"))
            for source, target, data in bwd_message
        ]
        return new_message

    def backward_state_evolution(self, message):
        if self.n_prev == 0:
            return []
        a_hat = self.posterior_a(message)
        fwd_message = filter_message(message, "fwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], direction="bwd"))
            for source, target, data in fwd_message
        ]
        return new_message


class BridgeVariable(Variable):
    n_next = 1
    n_prev = 1


class FinalVariable(Variable):
    n_next = 0
    n_prev = 1

def inv(v):
    """Numerically safe inverse"""
    return 1 / np.maximum(v, 1e-20)


class Factor(ReprMixin):

    def __add__(self, other):
        from .models.factor_algebra import FactorDAG
        return FactorDAG(self) + other

    def __matmul__(self, other):
        from .models.factor_algebra import FactorDAG
        return FactorDAG(self) @ other

    def check_message(self, message):
        for source, target, data in message:
            if (target != self):
                raise ValueError(f"target {target} is not the instance {self}")
            if not isinstance(source, Variable):
                raise ValueError(f"source {source} is not a Variable")
        n_next = len(filter_message(message, "bwd"))
        n_prev = len(filter_message(message, "fwd"))
        if (self.n_next != n_next):
            raise ValueError(
                f"number of next variables : expected {self.n_next} got {n_next}")
        if (self.n_prev != n_prev):
            raise ValueError(
                f"number of prev variables : expected {self.n_prev} got {n_prev}")


    def forward_message(self, message):
        if self.n_next == 0:
            return []
        fwd_posterior = self.forward_posterior(message)
        bwd_message = filter_message(message, "bwd")
        assert len(fwd_posterior) == len(bwd_message)
        new_message = [
            (target, source,
             dict(a=inv(v) - data["a"], b=r*inv(v) - data["b"], direction="fwd"))
            for (r, v), (source, target, data) in zip(fwd_posterior, bwd_message)
        ]
        return new_message

    def backward_message(self, message):
        if self.n_prev == 0:
            return []
        bwd_posterior = self.backward_posterior(message)
        fwd_message = filter_message(message, "fwd")
        assert len(bwd_posterior) == len(fwd_message)
        new_message = [
            (target, source,
             dict(a=inv(v) - data["a"], b=r*inv(v) - data["b"], direction="bwd"))
            for (r, v), (source, target, data) in zip(bwd_posterior, fwd_message)
        ]
        return new_message

    def forward_state_evolution(self, message):
        if self.n_next == 0:
            return []
        fwd_error = self.forward_error(message)
        bwd_message = filter_message(message, "bwd")
        assert len(fwd_error) == len(bwd_message)
        new_message = [
            (target, source,
             dict(a=inv(v) - data["a"], direction="fwd"))
            for v, (source, target, data) in zip(fwd_error, bwd_message)
        ]
        return new_message

    def backward_state_evolution(self, message):
        if self.n_prev == 0:
            return []
        bwd_error = self.backward_error(message)
        fwd_message = filter_message(message, "fwd")
        assert len(bwd_error) == len(fwd_message)
        new_message = [
            (target, source,
             dict(a=inv(v) - data["a"], direction="bwd"))
            for v, (source, target, data) in zip(bwd_error, fwd_message)
        ]
        return new_message


class Channel(Factor):
    n_next = 1
    n_prev = 1

    def _parse_message(self, message):
        # for x=channel(z)
        fwd_message = filter_message(message, "fwd")
        bwd_message = filter_message(message, "bwd")
        assert len(fwd_message) == 1 and len(bwd_message) == 1
        _, _, zdata = fwd_message[0]  # prev variable z send fwd message
        _, _, xdata = bwd_message[0]  # next variable x send bwd message
        return zdata, xdata
    
    def _parse_endpoints(self, message, direction):
        dir_message = filter_message(message, direction)
        assert len(dir_message) == 1
        source, target, _ = dir_message[0]
        return source, target

    def forward_posterior(self, message):
        zdata, xdata = self._parse_message(message)
        az, bz, ax, bx = zdata["a"], zdata["b"], xdata["a"], xdata["b"]
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        return [(rx, vx)]

    def backward_posterior(self, message):
        zdata, xdata = self._parse_message(message)
        az, bz, ax, bx = zdata["a"], zdata["b"], xdata["a"], xdata["b"]
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        return [(rz, vz)]

    def compute_forward_error(self, az, ax, tau):
        def variance(bz, bx):
            rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
            return vx
        error = self.beliefs_measure(az, ax, tau, f=variance)
        return error

    def compute_backward_error(self, az, ax, tau):
        def variance(bz, bx):
            rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
            return vz
        error = self.beliefs_measure(az, ax, tau, f=variance)
        return error

    def forward_error(self, message):
        zdata, xdata = self._parse_message(message)
        az, ax, tau = zdata["a"], xdata["a"], zdata["tau"]
        error = self.compute_forward_error(az, ax, tau)
        return [error]

    def backward_error(self, message):
        zdata, xdata = self._parse_message(message)
        az, ax, tau = zdata["a"], xdata["a"], zdata["tau"]
        error = self.compute_backward_error(az, ax, tau)
        return [error]


class Likelihood(Factor):
    n_next = 0
    n_prev = 1

    def _parse_message(self, message):
        source, target, data = message[0]
        assert len(message) == 1 and data["direction"] == "fwd"
        return data

    def _parse_endpoints(self, message):
        source, target, data = message[0]
        assert len(message) == 1 and data["direction"] == "fwd"
        return source, target

    def backward_posterior(self, message):
        data = self._parse_message(message)
        az, bz = data["a"], data["b"]
        rz, vz = self.compute_backward_posterior(az, bz, self.y)
        return [(rz, vz)]

    def compute_backward_error(self, az, tau):
        def variance(bz, y):
            rz, vz = self.compute_backward_posterior(az, bz, y)
            return vz
        error = self.beliefs_measure(az, tau, f=variance)
        return error

    def backward_error(self, message):
        data = self._parse_message(message)
        az, tau = data["a"], data["tau"]
        error = self.compute_backward_error(az, tau)
        return [error]


class Prior(Factor):
    n_next = 1
    n_prev = 0

    def _parse_message(self, message):
        source, target, data = message[0]
        assert len(message) == 1 and data["direction"] == "bwd"
        return data

    def _parse_endpoints(self, message):
        source, target, data = message[0]
        assert len(message) == 1 and data["direction"] == "bwd"
        return source, target

    def forward_posterior(self, message):
        data = self._parse_message(message)
        ax, bx = data["a"], data["b"]
        rx, vx = self.compute_forward_posterior(ax, bx)
        return [(rx, vx)]

    def compute_forward_error(self, ax):
        def variance(bx):
            rx, vx = self.compute_forward_posterior(ax, bx)
            return vx
        error = self.beliefs_measure(ax, f=variance)
        return error

    def forward_error(self, message):
        data = self._parse_message(message)
        ax = data["a"]
        error = self.compute_forward_error(ax)
        return [error]


class Ensemble(ReprMixin):
    pass


class Input(ReprMixin):
    pass

class Student(ReprMixin):
    pass


class Teacher(ReprMixin):
    pass


class Model():
    pass

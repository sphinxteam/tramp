"""
Base classes.
"""

import numpy as np
from scipy.integrate import quad, dblquad
import logging


class ReprMixin():
    _repr_initialized = False

    def repr_init(self, pad=None, reinit=False):
        if reinit or not self._repr_initialized:
            self._repr_kwargs = self.__dict__.copy()
            self._repr_pad = pad
            self._repr_initialized = True

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


def inv(v):
    """Numerically safe inverse"""
    return 1 / np.maximum(v, 1e-20)


AMAX = 1e+11
AMIN = 1e-11


def compute_a_new(v, a):
    "Compute a_new and b_new ensuring that a_new is between AMIN and AMAX"
    a_new = np.clip(inv(v) - a, AMIN, AMAX)
    return a_new


def compute_ab_new(r, v, a, b):
    "Compute a_new and b_new ensuring that a_new is between AMIN and AMAX"
    a_new = np.clip(inv(v) - a, AMIN, AMAX)
    v_inv = (a + a_new)
    b_new = r * v_inv - b
    return a_new, b_new


class Variable(ReprMixin):

    def __init__(self, n_prev, n_next, id, dtype=float):
        self.n_prev = n_prev
        self.n_next = n_next
        self.id = id
        self.repr_init()
        self.dtype = dtype

    def __add__(self, other):
        from .models.dag_algebra import DAG
        return DAG(self) + other

    def __matmul__(self, other):
        from .models.dag_algebra import DAG
        return DAG(self) @ other

    def math(self):
        return r"$" + self.id + r"$"

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

    def _parse_message_ab(self, message):
        # prev factor k send fwd message
        k_message = filter_message(message, "fwd")
        assert len(k_message) == self.n_prev
        ak = [data["a"] for source, target, data in k_message]
        bk = [data["b"] for source, target, data in k_message]
        k_source = [source for source, target, data in k_message]
        if self.n_prev == 1:
            ak = ak[0]
            bk = bk[0]
            k_source = k_source[0]
        # next factor l send bwd message
        l_message = filter_message(message, "bwd")
        assert len(l_message) == self.n_next
        al = [data["a"] for source, target, data in l_message]
        bl = [data["b"] for source, target, data in l_message]
        l_source = [source for source, target, data in l_message]
        if self.n_next == 1:
            al = al[0]
            bl = bl[0]
            l_source = l_source[0]
        return k_source, l_source, ak, bk, al, bl

    def _parse_message_a(self, message):
        # prev factor k send fwd message
        k_message = filter_message(message, "fwd")
        assert len(k_message) == self.n_prev
        ak = [data["a"] for source, target, data in k_message]
        k_source = [source for source, target, data in k_message]
        if self.n_prev == 1:
            ak = ak[0]
            k_source = k_source[0]
        # next factor l send bwd message
        l_message = filter_message(message, "bwd")
        assert len(l_message) == self.n_next
        al = [data["a"] for source, target, data in l_message]
        l_source = [source for source, target, data in l_message]
        if self.n_next == 1:
            al = al[0]
            l_source = l_source[0]
        return k_source, l_source, ak, al

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
        # next factor l send bwd message
        l_message = filter_message(message, "bwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], b=b_hat - data["b"], direction="fwd"))
            for source, target, data in l_message
        ]
        return new_message

    def backward_message(self, message):
        if self.n_prev == 0:
            return []
        a_hat, b_hat = self.posterior_ab(message)
        # prev factor k send fwd message
        k_message = filter_message(message, "fwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], b=b_hat - data["b"], direction="bwd"))
            for source, target, data in k_message
        ]
        return new_message

    def forward_state_evolution(self, message):
        if self.n_next == 0:
            return []
        a_hat = self.posterior_a(message)
        # next factor l send bwd message
        l_message = filter_message(message, "bwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], direction="fwd"))
            for source, target, data in l_message
        ]
        return new_message

    def backward_state_evolution(self, message):
        if self.n_prev == 0:
            return []
        a_hat = self.posterior_a(message)
        # prev factor k send fwd message
        k_message = filter_message(message, "fwd")
        new_message = [
            (target, source,
             dict(a=a_hat - data["a"], direction="bwd"))
            for source, target, data in k_message
        ]
        return new_message


class SIMOVariable(Variable):

    def __init__(self, n_next, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=n_next, dtype=dtype, id=id)


class MISOVariable(Variable):

    def __init__(self, n_prev, dtype=float, id=None):
        super().__init__(n_prev=n_prev, n_next=1, dtype=dtype, id=id)


class SISOVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=1, dtype=dtype, id=id)

    def forward_message(self, message):
        "pass message from previous factor k to next factor l"
        k_source, l_source, ak, bk, al, bl = self._parse_message_ab(message)
        new_message = [(self, l_source, dict(a=ak, b=bk, direction="fwd"))]
        return new_message

    def backward_message(self, message):
        "pass message from next factor l to previous factor k"
        k_source, l_source, ak, bk, al, bl = self._parse_message_ab(message)
        new_message = [(self, k_source, dict(a=al, b=bl, direction="bwd"))]
        return new_message

    def forward_state_evolution(self, message):
        "pass message from previous factor k to next factor l"
        k_source, l_source, ak, al = self._parse_message_a(message)
        new_message = [(self, l_source, dict(a=ak, direction="fwd"))]
        return new_message

    def backward_state_evolution(self, message):
        "pass message from next factor l to previous factor k"
        k_source, l_source, ak, al = self._parse_message_a(message)
        new_message = [(self, k_source, dict(a=al, direction="bwd"))]
        return new_message


class MILeafVariable(Variable):

    def __init__(self, n_prev, dtype=float, id=None):
        super().__init__(n_prev=n_prev, n_next=0, dtype=dtype, id=id)


class SILeafVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=1, n_next=0, dtype=dtype, id=id)


class MORootVariable(Variable):

    def __init__(self, n_next, dtype=float, id=None):
        super().__init__(n_prev=0, n_next=n_next, dtype=dtype, id=id)


class SORootVariable(Variable):

    def __init__(self, dtype=float, id=None):
        super().__init__(n_prev=0, n_next=1, dtype=dtype, id=id)


class Factor(ReprMixin):

    def __add__(self, other):
        from .models.dag_algebra import DAG
        return DAG(self) + other

    def __matmul__(self, other):
        from .models.dag_algebra import DAG
        return DAG(self) @ other

    def check_message(self, message):
        for source, target, data in message:
            if (target != self):
                raise ValueError(f"target {target} is not the instance {self}")
            if not isinstance(source, Variable):
                raise ValueError(f"source {source} is not a Variable")
        n_prev = len(filter_message(message, "fwd"))
        n_next = len(filter_message(message, "bwd"))
        if self.n_prev != n_prev:
            raise ValueError(f"expected n_prev={self.n_prev} got {n_prev}")
        if self.n_next != n_next:
            raise ValueError(f"expected n_next={self.n_next} got {n_next}")

    def _parse_message_ab(self, message):
        # prev variable z send fwd message
        z_message = filter_message(message, "fwd")
        assert len(z_message) == self.n_prev
        az = [data["a"] for source, target, data in z_message]
        bz = [data["b"] for source, target, data in z_message]
        z_source = [source for source, target, data in z_message]
        if self.n_prev == 1:
            az = az[0]
            bz = bz[0]
            z_source = z_source[0]
        # next variable x send bwd message
        x_message = filter_message(message, "bwd")
        assert len(x_message) == self.n_next
        ax = [data["a"] for source, target, data in x_message]
        bx = [data["b"] for source, target, data in x_message]
        x_source = [source for source, target, data in x_message]
        if self.n_next == 1:
            ax = ax[0]
            bx = bx[0]
            x_source = x_source[0]
        return z_source, x_source, az, bz, ax, bx

    def _parse_message_a(self, message):
        # prev variable z send fwd message
        z_message = filter_message(message, "fwd")
        assert len(z_message) == self.n_prev
        az = [data["a"] for source, target, data in z_message]
        tau = [data["tau"] for source, target, data in z_message]
        z_source = [source for source, target, data in z_message]
        if self.n_prev == 1:
            az = az[0]
            tau = tau[0]
            z_source = z_source[0]
        # next variable x send bwd message
        x_message = filter_message(message, "bwd")
        assert len(x_message) == self.n_next
        ax = [data["a"] for source, target, data in x_message]
        x_source = [source for source, target, data in x_message]
        if self.n_next == 1:
            ax = ax[0]
            x_source = x_source[0]
        return z_source, x_source, az, ax, tau

    def forward_message(self, message):
        if self.n_next == 0:
            return []
        z_source, x_source, az, bz, ax, bx = self._parse_message_ab(message)
        if self.n_prev == 0:
            ax_new, bx_new = self.compute_forward_message(ax, bx)
        else:
            ax_new, bx_new = self.compute_forward_message(az, bz, ax, bx)
        if self.n_next == 1:
            new_message = [(
                self, x_source, dict(a=ax_new, b=bx_new, direction="fwd")
            )]
        else:
            new_message = [
                (self, source, dict(a=a, b=b, direction="fwd"))
                for a, b, source in zip(ax_new, bx_new, x_source)
            ]
        return new_message

    def backward_message(self, message):
        if self.n_prev == 0:
            return []
        z_source, x_source, az, bz, ax, bx = self._parse_message_ab(message)
        if self.n_next == 0:
            az_new, bz_new = self.compute_backward_message(az, bz)
        else:
            az_new, bz_new = self.compute_backward_message(az, bz, ax, bx)
        if self.n_prev == 1:
            new_message = [(
                self, z_source, dict(a=az_new, b=bz_new, direction="bwd")
            )]
        else:
            new_message = [
                (self, source, dict(a=a, b=b, direction="bwd"))
                for a, b, source in zip(az_new, bz_new, z_source)
            ]
        return new_message

    def forward_state_evolution(self, message):
        if self.n_next == 0:
            return []
        z_source, x_source, az, ax, tau = self._parse_message_a(message)
        if self.n_prev == 0:
            ax_new = self.compute_forward_state_evolution(ax)
        else:
            ax_new = self.compute_forward_state_evolution(az, ax, tau)
        if self.n_next == 1:
            new_message = [(
                self, x_source, dict(a=ax_new, direction="fwd")
            )]
        else:
            new_message = [
                (self, source, dict(a=a, direction="fwd"))
                for a, source in zip(ax_new, x_source)
            ]
        return new_message

    def backward_state_evolution(self, message):
        if self.n_prev == 0:
            return []
        z_source, x_source, az, ax, tau = self._parse_message_a(message)
        if self.n_next == 0:
            az_new = self.compute_backward_state_evolution(az, tau)
        else:
            az_new = self.compute_backward_state_evolution(az, ax, tau)
        if self.n_prev == 1:
            new_message = [(
                self, z_source, dict(a=az_new, direction="bwd")
            )]
        else:
            new_message = [
                (self, source, dict(a=a, direction="bwd"))
                for a, source in zip(az_new, z_source)
            ]
        return new_message


class Channel(Factor):
    n_next = 1
    n_prev = 1

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new, bx_new = compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new, bz_new = compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau):
        vx = self.compute_forward_error(az, ax, tau)
        ax_new = compute_a_new(vx, ax)
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau):
        vz = self.compute_backward_error(az, ax, tau)
        az_new = compute_a_new(vz, az)
        return az_new

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

    def free_energy(self, az, ax, tau):
        def log_partition(bz, bx):
            return self.log_partition(az, bz, ax, bx)
        A = self.beliefs_measure(az, ax, tau, f=log_partition)
        return A


class Likelihood(Factor):
    n_next = 0
    n_prev = 1

    def compute_backward_message(self, az, bz):
        rz, vz = self.compute_backward_posterior(az, bz, self.y)
        az_new, bz_new = compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, tau):
        vz = self.compute_backward_error(az, tau)
        az_new = compute_a_new(vz, az)
        return az_new

    def compute_backward_error(self, az, tau):
        def variance(bz, y):
            rz, vz = self.compute_backward_posterior(az, bz, y)
            return vz
        error = self.beliefs_measure(az, tau, f=variance)
        return error

    def free_energy(self, az, tau):
        def log_partition(bz, y):
            return self.log_partition(az, bz, y)
        A = self.beliefs_measure(az, tau, f=log_partition)
        return A


class Prior(Factor):
    n_next = 1
    n_prev = 0

    def compute_forward_message(self, ax, bx):
        rx, vx = self.compute_forward_posterior(ax, bx)
        ax_new, bx_new = compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_forward_state_evolution(self, ax):
        vx = self.compute_forward_error(ax)
        ax_new = compute_a_new(vx, ax)
        return ax_new

    def compute_forward_error(self, ax):
        def variance(bx):
            rx, vx = self.compute_forward_posterior(ax, bx)
            return vx
        error = self.beliefs_measure(ax, f=variance)
        return error

    def free_energy(self, ax):
        def log_partition(bx):
            return self.log_partition(ax, bx)
        A = self.beliefs_measure(ax, f=log_partition)
        return A


class Ensemble(ReprMixin):
    pass


class Model(ReprMixin):
    pass

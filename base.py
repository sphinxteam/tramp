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


class Variable(ReprMixin):

    def __init__(self, id, n_prev, n_next):
        self.id = id
        self.n_prev = n_prev
        self.n_next = n_next
        self.repr_init()

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


class Factor(ReprMixin):

    AMAX = 1e+11
    AMIN = 1e-11

    def reset_precision_bounds(AMIN, AMAX):
        self.AMIN = AMIN
        self.AMAX = AMAX

    def compute_a_new(self, v, a):
        "Compute a_new and b_new ensuring that a_new is between AMIN and AMAX"
        a_new = np.clip(inv(v) - a, self.AMIN, self.AMAX)
        return a_new

    def compute_ab_new(self, r, v, a, b):
        "Compute a_new and b_new ensuring that a_new is between AMIN and AMAX"
        a_new = np.clip(inv(v) - a, self.AMIN, self.AMAX)
        v_inv = (a + a_new)
        b_new = r * v_inv - b
        return a_new, b_new

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

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ab_new = [
            self.compute_ab_new(rk, vk, ak, bk)
            for rk, vk, ak, bk in zip(rx, vx, ax, bx)
        ]
        ax_new = [a for a, b in ab_new]
        bx_new = [b for a, b in ab_new]
        return ax_new, bx_new

    def compute_forward_state_evolution(self, az, ax, tau):
        vx = self.compute_forward_error(az, ax, tau)
        ax_new = [self.compute_a_new(vk, ak) for vk, ak in zip(vx, ax)]
        return ax_new

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        ab_new = [
            self.compute_ab_new(rk, vk, ak, bk)
            for rk, vk, ak, bk in zip(rz, vz, az, bz)
        ]
        az_new = [a for a, b in ab_new]
        bz_new = [b for a, b in ab_new]
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, ax, tau):
        vz = self.compute_backward_error(az, ax, tau)
        az_new = [self.compute_a_new(vk, ak) for vk, ak in zip(vz, az)]
        return az_new

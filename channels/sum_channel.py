import numpy as np
from ..base import Factor, inv


class SumChannel(Factor):
    n_next = 1

    def __init__(self, n_prev):
        self.n_prev = n_prev
        self.repr_init()

    def sample(self, *Zs):
        if len(Zs) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} arrays")
        X = sum(Zs)
        return X

    def math(self):
        return r"$\sum$"

    def second_moment(self, *taus):
        if len(taus) != self.n_prev:
            raise ValueError(f"expect {self.n_prev} arrays")
        tau = sum(taus)
        return tau

    def _parse_message(self, message):
        # prev variables z = {zk} send fwd message
        z_message = filter_message(message, "fwd")
        # next variable x send bwd message
        x_message = filter_message(message, "bwd")
        assert len(z_message) == self.n_prev and len(x_message) == 1
        z_sources = [source for source, target, data in z_message]
        z_targets = [target for source, target, data in z_message]
        assert all(target is self for target in z_targets)
        z_data = [data for source, target, data in z_message]
        x_source, x_target, x_data = x_message[0]
        assert x_target is self
        return z_sources, z_data, x_source, x_data

    def forward_message(self, message):
        # fwd message to x; for x = sum(z)
        z_sources, z_data, x_source, x_data = self._parse_message(message)
        v_bar = sum(1 / data["a"] for data in z_data)
        r_bar = sum(data["b"] / data["a"] for data in z_data)
        new_data = dict(a=inv(v_bar), b=r_bar * inv(v_bar), direction="fwd")
        new_message = [(self, x_source, new_data)]
        return new_message

    def backward_message(self, message):
        # bwd message to z = {zk}; for x = sum(z)
        z_sources, z_data, x_source, x_data = self._parse_message(message)
        v_bar = sum(1 / data["a"] for data in z_data)
        r_bar = sum(data["b"] / data["a"] for data in z_data)
        vx = 1 / x_data["a"]
        rx = x_data["b"] / x_data["a"]
        vk = [vx + v_bar - 1 / data["a"] for data in z_data]
        rk = [rx - r_bar + data["b"] / data["a"] for data in z_data]
        new_z_data = [
            dict(a=inv(v), b=r * inv(v), direction="bwd")
            for v, r in zip(vk, rk)
        ]
        new_message = [
            (self, z_source, data)
            for z_source, data in zip(z_sources, new_z_data)
        ]
        return new_message

    def forward_state_evolution(self, message):
        # fwd message to x; for x = sum(z)
        z_sources, z_data, x_source, x_data = self._parse_message(message)
        v_bar = sum(1 / data["a"] for data in z_data)
        new_data = dict(a=inv(v_bar), direction="fwd")
        new_message = [(self, x_source, new_data)]
        return new_message

    def backward_state_evolution(self, message):
        # bwd message to z = {zk}; for x = sum(z)
        z_sources, z_data, x_source, x_data = self._parse_message(message)
        v_bar = sum(1 / data["a"] for data in z_data)
        vx = 1 / x_data["a"]
        vk = [vx + v_bar - 1 / data["a"] for data in z_data]
        new_z_data = [
            dict(a=inv(v), direction="bwd") for v in vk
        ]
        new_message = [
            (self, z_source, data)
            for z_source, data in zip(z_sources, new_z_data)
        ]
        return new_message

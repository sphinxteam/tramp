import numpy as np
from ..base import Channel


def check_rotation(R):
    if (R.shape[0] != R.shape[1]):
        raise ValueError(f"R.shape = {R.shape}")
    N = R.shape[0]
    if not np.allclose(R @ R.T, np.identity(N)):
        raise ValueError("R not a rotation")


class RotationChannel(Channel):
    def __init__(self, R):
        check_rotation(R)
        self.N = R.shape[0]
        self.repr_init()
        self.R = R

    def sample(self, Z):
        X = self.R @ Z
        return X

    def math(self):
        return r"$R$"

    def second_moment(self, tau):
        return tau

    def forward_message(self, message):
        # x = R z, x source variable of bwd message
        az, bz, ax, bx = self._parse_message_ab(message)
        source, target = self._parse_endpoints(message, "bwd")
        new_data = dict(a=az, b=self.R @ bz, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_message(self, message):
        # z = R.T x, z source variable of fwd message
        az, bz, ax, bx = self._parse_message_ab(message)
        source, target = self._parse_endpoints(message, "fwd")
        new_data = dict(a=ax, b=self.R.T @ bx, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

    def forward_state_evolution(self, message):
        # x = R z, x source variable of bwd message
        az, ax = self._parse_message_a(message)
        source, target = self._parse_endpoints(message, "bwd")
        new_data = dict(a=az, direction="fwd")
        new_message = [(target, source, new_data)]
        return new_message

    def backward_state_evolution(self, message):
        # z = R.T x, z source variable of fwd message
        az, ax = self._parse_message_a(message)
        source, target = self._parse_endpoints(message, "fwd")
        new_data = dict(a=ax, direction="bwd")
        new_message = [(target, source, new_data)]
        return new_message

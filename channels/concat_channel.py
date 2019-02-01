from ..base import Factor, filter_message
import numpy as np

class ConcatChannel(Factor):
    n_next = 1
    n_prev = 2

    def __init__(self, N1, N2, axis=0):
        self.N1 = N1
        self.N2 = N2
        self.N = self.N1 + self.N2
        self.axis = axis
        self.repr_init()

    def sample(self, Z1, Z2):
        if (Z1.shape[self.axis] != self.N1):
            raise ValueError(
                f"Expected Z1 array of size {self.N1} along axis {self.axis}"
            )
        if (Z2.shape[self.axis] != self.N2):
            raise ValueError(
                f"Expected Z2 array of size {self.N2} along axis {self.axis}"
            )
        X = np.concatenate((Z1, Z2), axis=self.axis)
        assert X.shape[self.axis] == self.N
        return X

    def math(self):
        return r"$\oplus$"

    def second_moment(self, tau1, tau2):
        tau = (self.N1 * tau1 + self.N2 * tau2) / self.N
        return tau

    def _parse_message(self, message):
        # for x=[z1, z2]
        fwd_message = filter_message(message, "fwd")
        bwd_message = filter_message(message, "bwd")
        assert len(fwd_message) == 2 and len(bwd_message) == 1
        _, _, z1data = fwd_message[0] # prev variable z1 send fwd message
        _, _, z2data = fwd_message[1] # prev variable z2 send fwd message
        _, _, xdata = bwd_message[0]  # next variable x  send bwd message
        return z1data, z2data, xdata

    def forward_posterior(self, message):
        # estimate x from x = [z1, z2]
        [(r1, v1), (r2, v2)] = self.backward_posterior(message)
        rx = np.concatenate((r1, r2), axis=self.axis)
        vx = (self.N1 * v1 + self.N2 * v2) / self.N
        return [(rx, vx)]

    def backward_posterior(self, message):
        # estimate z1, z2 from x = [z1, z2]
        z1data, z2data, xdata = self._parse_message(message)
        az1, bz1 = z1data["a"], z1data["b"]
        az2, bz2 = z2data["a"], z2data["b"]
        ax, bx = xdata["a"], xdata["b"]
        assert bz1.shape[self.axis]==self.N1
        assert bz2.shape[self.axis]==self.N2
        assert bx.shape[self.axis]==self.N
        bx1 = np.take(bx, range(self.N1), axis=self.axis)
        bx2 = np.take(bx, range(self.N1, self.N), axis=self.axis)
        r1 = (bz1 + bx1) / (az1 + ax)
        r2 = (bz2 + bx2) / (az2 + ax)
        v1 = 1 / (az1 + ax)
        v2 = 1 / (az2 + ax)
        return [(r1, v1), (r2, v2)]

    def forward_error(self, message):
        [v1, v2] = self.backward_error(message)
        vx = (self.N1 * v1 + self.N2 * v2) / self.N
        return [vx]

    def backward_error(self, message):
        z1data, z2data, xdata = self._parse_message(message)
        az1, az2, ax = z1data["a"], z2data["a"], xdata["a"]
        v1 = 1 / (az1 + ax)
        v2 = 1 / (az2 + ax)
        return [v1, v2]

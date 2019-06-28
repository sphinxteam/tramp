import numpy as np
from .activation_channel import ActivationChannel

class TanhChannel(ActivationChannel):
    def __init__(self):
        super().__init__(func = np.tanh)

# import numpy as np
# from ..base import Channel
# from ..utils.integration import gaussian_measure
# from scipy.integrate import quad
#
#
# class TanhChannel(Channel):
#     def __init__(self):
#         self.repr_init()
#
#     def sample(self, Z):
#         X = np.tanh(Z)
#         return X
#
#     def math(self):
#         return r"$\mathrm{tanh}$"
#
#     def second_moment(self, tau):
#         tau_x = gaussian_measure(0, np.sqrt(tau), f = lambda z: np.tanh(z)**2)
#         return tau_x
#
#     def compute_forward_posterior(self, az, bz, ax, bx):
#         # estimate x from x = tanh(z)
#         raise NotImplementedError
#         return rx, vx
#
#     def compute_backward_posterior(self, az, bz, ax, bx):
#         # estimate z from x = tanh(z)
#         raise NotImplementedError
#         return rz, vz
#
#     def beliefs_measure(self, az, ax, tau, f):
#         raise NotImplementedError
#
#     def measure(self, f, zmin, zmax):
#         def integrand(z):
#             return f(z, np.tanh(z))
#         return quad(integrand, zmin, zmax)[0]

from ..base_channel import SIFactor


class DuplicateChannel(SIFactor):

    def __init__(self, n_next):
        self.n_next = n_next
        self.repr_init()

    def sample(self, Z):
        return (Z,) * self.n_next

    def math(self):
        return r"$\delta$"

    def second_moment(self, tau_z):
        return (tau_z,) * self.n_next

    def compute_forward_posterior(self, az, bz, ax, bx):
        "estimate x = {xk} from (xk = z for all k)"
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        rx = [rz] * self.n_next
        vx = [vz] * self.n_next
        return rx, vx

    def compute_backward_posterior(self, az, bz, ax, bx):
        "estimate z from (xk = z for all k)"
        a = az + sum(ax)
        b = bz + sum(bx)
        rz = b / a
        vz = 1. / a
        return rz, vz

    def compute_forward_error(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        vx = [vz] * self.n_next
        return vx

    def compute_backward_error(self, az, ax, tau_z):
        a = az + sum(ax)
        vz = 1. / a
        return vz

    def compute_log_partition(self, az, bz, ax, bx):
        a = az + sum(ax)
        b = bz + sum(bx)
        logZ = 0.5 * np.sum(b**2 / a + np.log(2 * np.pi / a))
        return logZ

    def compute_free_energy(self, az, ax, tau_z):
        raise NotImplementedError

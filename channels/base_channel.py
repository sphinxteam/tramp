from ..base import Factor


class Channel(Factor):
    n_next = 1
    n_prev = 1

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        vx = self.compute_forward_error(az, ax, tau_z)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        az_new = self.compute_a_new(vz, az)
        return az_new

    def compute_forward_error(self, az, ax, tau_z):
        def variance(bz, bx):
            rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
            return vx
        error = self.beliefs_measure(az, ax, tau_z, f=variance)
        return error

    def compute_backward_error(self, az, ax, tau_z):
        def variance(bz, bx):
            rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
            return vz
        error = self.beliefs_measure(az, ax, tau_z, f=variance)
        return error

    def free_energy(self, az, ax, tau_z):
        def log_partition(bz, bx):
            return self.log_partition(az, bz, ax, bx)
        A = self.beliefs_measure(az, ax, tau_z, f=log_partition)
        return A

class SIFactor(Factor):
    n_prev = 1

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        vx = self.compute_forward_error(az, ax, tau_z)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new


class SOFactor(Factor):
    n_next = 1

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        az_new = self.compute_a_new(vz, az)
        return az_new


class MatrixFactorization(SOFactor):
    n_prev = 2

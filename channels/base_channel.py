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

    def compute_forward_overlap(self, az, ax, tau_z):
        vx = self.compute_forward_error(az, ax, tau_z)
        tau_x = self.second_moment(tau_z)
        mx = tau_x - vx
        return mx

    def compute_backward_overlap(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        mz = tau_z - vz
        return mz

    def free_energy(self, az, ax, tau_z):
        def log_partition(bz, bx):
            return self.log_partition(az, bz, ax, bx)
        A = self.beliefs_measure(az, ax, tau_z, f=log_partition)
        return A

    def mutual_information(self, az, ax, tau_z, alpha = 1):
        tau_x = self.second_moment(tau_z)
        A = self.free_energy(az, ax, tau_z)
        I = 0.5*(az*tau_z + alpha*ax*tau_x) - A + 0.5*np.log(2*pi*tau_z/np.e)
        return A

class SIFactor(Factor):
    n_prev = 1

    def compute_backward_message(self, az, bz, ax, bx):
        rz, vz = self.compute_backward_posterior(az, bz, ax, bx)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        az_new = self.compute_a_new(vz, az)
        return az_new

    def compute_backward_overlap(self, az, ax, tau_z):
        vz = self.compute_backward_error(az, ax, tau_z)
        mz = tau_z - vz
        return mz


class SOFactor(Factor):
    n_next = 1

    def compute_forward_message(self, az, bz, ax, bx):
        rx, vx = self.compute_forward_posterior(az, bz, ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_forward_state_evolution(self, az, ax, tau_z):
        vx = self.compute_forward_error(az, ax, tau_z)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new

    def compute_forward_overlap(self, az, ax, tau_z):
        vx = self.compute_forward_error(az, ax, tau_z)
        tau_x = self.second_moment(tau_z)
        mx = tau_x - vx
        return mx


class MatrixFactorization(SOFactor):
    n_prev = 2

from ..base import Factor
from scipy.optimize  import root


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

    def compute_free_energy(self, az, ax, tau_z):
        def log_partition(bz, bx):
            return self.compute_log_partition(az, bz, ax, bx)
        A = self.beliefs_measure(az, ax, tau_z, f=log_partition)
        return A

    def get_alpha(self):
        return getattr(self, 'alpha', 1)

    def compute_mutual_information(self, az, ax, tau_z):
        alpha = self.get_alpha()
        tau_x = self.second_moment(tau_z)
        A = self.compute_free_energy(az, ax, tau_z)
        I = 0.5*(az*tau_z + alpha*ax*tau_x) - A + 0.5*np.log(2*np.pi*tau_z/np.e)
        return I

    def compute_precision(self, vz, vx, tau_z):
        def f(a):
            az, ax = a
            fz = self.compute_backward_error(az, ax, tau_z) - vz
            fx = self.compute_forward_error(az, ax, tau_z) - vx
            return fz, fx
        x0 = 1/vz, 1/vx
        sol = root(f, x0=x0, method='hybr')
        az, ax = sol.x
        return az, ax

    def compute_dual_mutual_information(self, vz, vx, tau_z):
        alpha = self.get_alpha()
        az, ax = self.compute_precision(vz, vx, tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        I_dual = I - 0.5*(az*vz + alpha*ax*vx)
        return I_dual

    def compute_dual_free_energy(self, mz, mx, tau_z):
        alpha = self.get_alpha()
        tau_x = self.second_moment(tau_z)
        vz = tau_z - mz
        vx = tau_x - mx
        az, ax = self.compute_precision(vz, vx, tau_z)
        A = self.compute_free_energy(az, ax, tau_z)
        A_dual = 0.5*(az*mz + alpha*ax*mx) - A
        return A_dual


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

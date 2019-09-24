from ..base import Factor


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

from ..base import Factor


class Likelihood(Factor):
    n_next = 0
    n_prev = 1

    def get_size(self, y):
        if y is None:
            size = None
        elif len(y.shape) == 1:
            size = y.shape[0]
        else:
            size = y.shape
        return size

    def compute_backward_message(self, az, bz):
        rz, vz = self.compute_backward_posterior(az, bz, self.y)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, tau_z):
        vz = self.compute_backward_error(az, tau_z)
        az_new = self.compute_a_new(vz, az)
        return az_new

    def compute_backward_error(self, az, tau_z):
        def variance(bz, y):
            rz, vz = self.compute_backward_posterior(az, bz, y)
            return vz
        error = self.beliefs_measure(az, tau_z, f=variance)
        return error

    def compute_backward_overlap(self, az, tau_z):
        vz = self.compute_backward_error(az, tau_z)
        mz = tau_z - vz
        return mz

    def compute_free_energy(self, az, tau_z):
        def log_partition(bz, y):
            return self.compute_log_partition(az, bz, y)
        A = self.beliefs_measure(az, tau_z, f=log_partition)
        return A

    def compute_mutual_information(self, az, tau_z):
        A = self.compute_free_energy(az, ax, tau_z)
        I = 0.5*az*tau_z - A + 0.5*np.log(2*pi*tau_z/np.e)
        return A

from ..base import Factor
from scipy.optimize  import root_scalar
import numpy as np


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

    def prior_log_partition_FG(self, tz_hat):
        return 0.5*np.log(2*np.pi/tz_hat)

    def backward_second_moment_FG(self, tz_hat):
        return 1/tz_hat

    def compute_backward_message(self, az, bz):
        rz, vz = self.compute_backward_posterior(az, bz, self.y)
        az_new, bz_new = self.compute_ab_new(rz, vz, az, bz)
        return az_new, bz_new

    def compute_backward_state_evolution_BO(self, az, tz0_hat):
        vz = self.compute_backward_v_BO(az, tz0_hat)
        az_new = self.compute_a_new(vz, az)
        return az_new

    def compute_backward_v_BO(self, az, tz0_hat):
        mz_hat = az - tz0_hat
        def v_func(bz, y): return self.scalar_backward_variance(az, bz, y)
        vz = self.b_measure(mz_hat, mz_hat, tz0_hat, v_func)
        return vz

    def compute_potential_BO(self, az, tz0_hat):
        mz_hat = az - tz0_hat
        def A_func(bz, y): return self.scalar_log_partition(az, bz, y)
        A = self.b_measure(mz_hat, mz_hat, tz0_hat, A_func)
        return A

    def compute_backward_state_evolution_RS(self, az, mz_hat, qz_hat,
                                           teacher, tz0_hat, tz0):
        vz, mz, qz = self.compute_backward_vmq_RS(
            az, mz_hat, qz_hat, teacher, tz0_hat
        )
        az_new, mz_hat_new, qz_hat_new = self.compute_a_mhat_qhat_new(
            vz, mz, qz, az, mz_hat, qz_hat, tz0
        )
        return az_new, mz_hat_new, qz_hat_new

    def compute_backward_vmq_RS(self, az, mz_hat, qz_hat,
                               teacher, tz0_hat):
        def v_func(bz, y): return self.scalar_backward_variance(az, bz, y)
        def r_func(bz, y): return self.scalar_backward_mean(az, bz, y)
        def q_func(bz, y): return self.scalar_backward_mean(az, bz, y)**2
        vz = teacher.b_measure(mz_hat, qz_hat, tz0_hat, v_func)
        mz = teacher.bz_measure(mz_hat, qz_hat, tz0_hat, r_func)
        qz = teacher.b_measure(mz_hat, qz_hat, tz0_hat, q_func)
        return vz, mz, qz

    def compute_potential_RS(self, az, mz_hat, qz_hat,
                             teacher, tz0_hat):
        def A_func(bz, y): return self.scalar_log_partition(az, bz, y)
        A = teacher.b_measure(mz_hat, qz_hat, tz0_hat, A_func)
        return A

    def compute_backward_state_evolution(self, az, tau_z):
        vz = self.compute_backward_error(az, tau_z)
        az_new = self.compute_a_new(vz, az)
        return az_new

    def compute_backward_error(self, az, tau_z):
        def v_func(bz, y): return self.scalar_backward_variance(az, bz, y)
        error = self.beliefs_measure(az, tau_z, v_func)
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
        "Note: returns H = mutual information I + noise entropy N"
        A = self.compute_free_energy(az, tau_z)
        H = 0.5*az*tau_z - A + 0.5*np.log(2*np.pi*tau_z/np.e)
        return H

    def compute_precision(self, vz, tau_z):
        def f(az):
            return self.compute_backward_error(az, tau_z) - vz
        sol = root_scalar(f, bracket=[1/tau_z, 1/vz], method='bisect')
        az = sol.root
        return az

    def compute_dual_mutual_information(self, vz, tau_z):
        az = self.compute_precision(vz, tau_z)
        I = self.compute_mutual_information(az, tau_z)
        I_dual = I - 0.5*az*vz
        return I_dual

    def compute_dual_free_energy(self, mz, tau_z):
        vz = tau_z - mz
        az = self.compute_precision(vz, tau_z)
        A = self.compute_free_energy(az, tau_z)
        A_dual = 0.5*az*mz - A
        return A_dual

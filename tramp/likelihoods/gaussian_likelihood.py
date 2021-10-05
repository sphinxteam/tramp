import numpy as np
from .base_likelihood import Likelihood
from ..utils.integration import gaussian_measure, gaussian_measure_2d_full
from ..beliefs import normal


class GaussianLikelihood(Likelihood):
    def __init__(self, y, var=1, y_name="y", isotropic=True):
        self.y_name = y_name
        self.size = self.get_size(y)
        self.var = var
        self.isotropic = isotropic
        self.repr_init()
        self.y = y
        self.sigma = np.sqrt(var)
        self.a = 1 / var
        self.b = None if y is None else y / var

    def sample(self, X):
        noise = self.sigma * np.random.standard_normal(X.shape)
        return X + noise

    def math(self):
        return r"$\mathcal{N}$"

    def scalar_backward_mean(self, az, bz, y):
        ay, by = self.a, self.a*y
        a = az + ay
        b = bz + by
        return b / a

    def scalar_backward_variance(self, az, bz, y):
        a = az + self.a
        return 1 / a

    def scalar_log_partition(self, az, bz, y):
        ay, by = self.a, self.a*y
        a = az + ay
        b = bz + by
        A = normal.A(a, b) - normal.A(ay, by)
        return A

    def compute_backward_posterior(self, az, bz, y):
        ay, by = self.a, self.a*y
        a = az + ay
        b = bz + by
        rz = b / a
        vz = 1 / a
        return rz, vz

    def compute_log_partition(self, az, bz, y):
        ay, by = self.a, self.a*y
        a = az + ay
        b = bz + by
        A = normal.A(a, b) - normal.A(ay, by)
        return A.mean()

    def compute_backward_error(self, az, tau_z):
        a = az + self.a
        vz = 1 / a
        return vz

    def compute_backward_v_BO(self, az, tz0_hat):
        a = az + self.a
        vz = 1 / a
        return vz

    def compute_backward_message(self, az, bz):
        az_new = self.a
        bz_new = self.b
        return az_new, bz_new

    def compute_backward_state_evolution(self, az, tau_z):
        az_new = self.a
        return az_new

    def compute_backward_state_evolution_BO(self, az, tau_z):
        az_new = self.a
        return az_new

    def b_measure(self, mz_hat, qz_hat, tz0_hat, f):
        tz0 = 1 / tz0_hat
        cov = np.array([
            [qz_hat + mz_hat**2 * tz0, mz_hat*tz0],
            [mz_hat*tz0, self.var + tz0]
        ])
        mean = np.array([0, 0])
        mu = gaussian_measure_2d_full(mean, cov, f)
        return mu

    def bz_measure(self, mz_hat, qz_hat, tz0_hat, f):
        az_star = (mz_hat / qz_hat) * mz_hat + tz0_hat
        ay = self.a

        def r_times_f(bz, y):
            bz_star = (mz_hat / qz_hat) * bz
            by = self.a*y
            r = (by + bz_star) / (ay + az_star)
            return r * f(bz, y)
        tz0 = 1 / tz0_hat
        cov = np.array([
            [qz_hat + mz_hat**2 * tz0, mz_hat*tz0],
            [mz_hat*tz0, self.var + tz0]
        ])
        mean = np.array([0, 0])
        mu = gaussian_measure_2d_full(mean, cov, r_times_f)
        return mu

    def beliefs_measure(self, az, tau_z, f):
        u_eff = np.maximum(0, az * tau_z - 1)
        cov = np.array([
            [u_eff*az, u_eff],
            [u_eff, self.var + tau_z]
        ])
        mu = gaussian_measure_2d_full(mean, cov, f)
        return mu

    def measure(self, y, f):
        return gaussian_measure(y, self.sigma, f)

    def compute_mutual_information(self, az, tau_z):
        "Note: returns H = mutual information I + noise entropy N"
        a = az + self.a
        I = 0.5*np.log(a*tau_z)
        N = 0.5*np.log(2*np.pi*np.e*self.var)
        H = I + N
        return H

    def compute_free_energy(self, az, tau_z):
        a = az + self.a
        A = 0.5*az*tau_z - 1 - 0.5*np.log(a*self.var)
        return A

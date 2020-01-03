import numpy as np
from ..base_channel import Channel
from tramp.ensembles import MarchenkoPasturEnsemble
import logging
logger = logging.getLogger(__name__)


class AnalyticalLinearChannel(Channel):
    def __init__(self, ensemble, name="W"):
        self.name = name
        self.alpha = ensemble.alpha
        self.repr_init()
        self.ensemble = ensemble

    def sample(self, Z):
        N = Z.shape[0]
        F = self.ensemble.generate(N)
        X = F @ Z
        return X

    def math(self):
        return r"$"+self.name+"$"

    def second_moment(self, tau_z):
        tau_x = tau_z * (self.ensemble.mean_spectrum / self.alpha)
        return tau_x

    def compute_n_eff(self, az, ax):
        "Effective number of parameters"
        if ax == 0:
            logger.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logger.info(f"az/ax=0 in {self} compute_n_eff")
            return min(1, self.alpha)
        gamma = ax / az
        n_eff = 1 - self.ensemble.eta_transform(gamma)
        return n_eff

    def compute_backward_error(self, az, ax, tau_z):
        if az==0:
            logger.info(f"az=0 in {self} compute_backward_error")
        az = np.maximum(1e-11, az)
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_error(self, az, ax, tau_z):
        if ax == 0:
            return self.ensemble.mean_spectrum / (self.alpha * az)
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / (self.alpha * ax)
        return vx

    def compute_mutual_information(self, az, ax, tau_z):
        gamma = ax / az
        S = self.ensemble.shannon_transform(gamma)
        I = 0.5*np.log(az*tau_z) + 0.5*S
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + self.alpha*ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A


class MarchenkoPasturChannel(AnalyticalLinearChannel):
    def __init__(self, alpha, name = "W"):
        ensemble=MarchenkoPasturEnsemble(alpha = alpha)
        super().__init__(ensemble = ensemble, name = name)

    def compute_precision(self, vz, vx, tau_z):
        ax = 1/vx - 1/vz
        az = (1 - self.alpha*ax*vx)/vz
        return az, ax

    def _compute_dual_mutual_information(self, vz, vx, tau_z):
        Iz = 0.5*np.log(tau_z/vz) - 0.5
        J = 0.5*self.alpha*(np.log(vz/vx) + vx/vz - 1)
        I_dual = J + Iz
        return I_dual

    def _compute_dual_free_energy(self, mz, mx, tau_z):
        tau_x = self.second_moment(tau_z)
        vz = tau_z - mz
        vx = tau_x - mx
        I_dual = self._compute_dual_mutual_information(vz, vx, tau_z)
        A_dual = I_dual - 0.5*np.log(2*np.pi*tau_z/np.e)
        return A_dual

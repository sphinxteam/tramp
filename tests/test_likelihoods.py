import unittest
from tramp.likelihoods import (
    GaussianLikelihood, SngLikelihood, AbsLikelihood, ModulusLikelihood
)
import numpy as np


def complex_dot(z1, z2):
    return np.real(np.conjugate(z1)*z2)

def explicit_real_integral(az, bz, y, likelihood):
    """
    Compute rz, vz for likelihood p(y|z) by integration for z real
    """
    def belief(z):
        L = -0.5 * az * (z**2) + bz * z
        return np.exp(L)
    def z_belief(z):
        return z*belief(z)
    def z2_belief(z):
        return (z**2)*belief(z)
    Z = likelihood.measure(y, belief)
    rz = likelihood.measure(y, z_belief) / Z
    z2 = likelihood.measure(y, z2_belief) / Z
    vz = z2 - rz**2
    return rz, vz


def explicit_complex_integral(az, bz, y, likelihood):
    """
    Compute rz, vz for likelihood p(y|z) by integration for z complex
    """
    def belief(z):
        L = -0.5 * az * complex_dot(z,z) + complex_dot(bz, z)
        return np.exp(L)
    def real_z_belief(z):
        return np.real(z)*belief(z)
    def imag_z_belief(z):
        return np.imag(z)*belief(z)
    def z2_belief(z):
        return complex_dot(z, z)*belief(z)
    Z = likelihood.measure(y, belief)
    real_rz = likelihood.measure(y, real_z_belief) / Z
    imag_rz = likelihood.measure(y, imag_z_belief) / Z
    rz = real_rz + imag_rz * 1j
    z2 = likelihood.measure(y, z2_belief) / Z
    vz = 0.5 * (z2 - complex_dot(rz, rz))
    return rz, vz


class likelihoodsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            dict(az=2.0, bz=2.0, tau = 1.0),
            dict(az=3.5, bz=1.3, tau = 0.5)
        ]

    def tearDown(self):
        pass

    def _test_function_posterior(self, likelihood, records, places=12):
        for record in records:
            az, bz = record["az"], record["bz"]
            y = float(likelihood.y)
            if isinstance(likelihood, ModulusLikelihood):
                explicit_integral = explicit_complex_integral
            else:
                explicit_integral = explicit_real_integral
            rz, vz = explicit_integral(az, bz, y, likelihood)
            rz_hat, vz_hat = likelihood.compute_backward_posterior(az, bz, y)
            rz_hat = float(rz_hat)
            msg = f"record={record} likelihood={likelihood}"
            self.assertAlmostEqual(rz, rz_hat, places=places, msg=msg)
            self.assertAlmostEqual(vz, vz_hat, places=places, msg=msg)

    def _test_function_proba(self, likelihood, records, places=12):
        for record in records:
            az, tau = record["az"], record["tau"]
            one = lambda bz, y: 1
            sum_proba = likelihood.beliefs_measure(az, tau, f=one)
            msg = f"record={record} likelihood={likelihood}"
            self.assertAlmostEqual(sum_proba, 1., places=places, msg=msg)

    def test_gaussian_posterior(self):
        likelihoods = [
            GaussianLikelihood(y = np.array([+1]), var=5.2),
            GaussianLikelihood(y = np.array([-1]), var=2.0)
        ]
        for likelihood in likelihoods:
            self._test_function_posterior(likelihood, self.records)

    def test_sng_posterior(self):
        likelihoods = [
            SngLikelihood(y = np.array([+1])),
            SngLikelihood(y = np.array([-1]))
        ]
        for likelihood in likelihoods:
            self._test_function_posterior(likelihood, self.records)

    def test_abs_posterior(self):
        likelihoods = [
            AbsLikelihood(y = np.array([10.4])),
            AbsLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            self._test_function_posterior(likelihood, self.records)

    def test_modulus_posterior(self):
        likelihoods = [
            ModulusLikelihood(y = np.array([10.4])),
            ModulusLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            # less precise, why ?
            self._test_function_posterior(likelihood, self.records, places=3)

    def test_sng_proba(self):
        likelihoods = [
            SngLikelihood(y = np.array([+1])),
            SngLikelihood(y = np.array([-1]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, self.records)

    def test_abs_proba(self):
        likelihoods = [
            AbsLikelihood(y = np.array([10.4])),
            AbsLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, self.records)

    def test_modulus_proba(self):
        likelihoods = [
            ModulusLikelihood(y = np.array([10.4])),
            ModulusLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, self.records, places=6)

if __name__ == "__main__":
    unittest.main()

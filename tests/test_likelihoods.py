import unittest
from tramp.likelihoods import SngLikelihood, AbsLikelihood, ModulusLikelihood
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import logging


def explicit_integral(az, bz, y, likelihood):
    """
    Compute rz, vz for likelihood p(y|z) by integration
    """
    raise NotImplementedError
    return rz, vz


class likelihoodsTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _test_function_posterior(self, likelihood, records):
        for record in records:
            az, bz = record["az"], record["bz"]
            y = likelihood.y
            rz, vz = explicit_integral(az, bz, y, likelihood)
            rz_hat, vz_hat = likelihood.compute_forward_posterior(az, bz, y)
            rz_hat = float(rz_hat)
            msg = f"record={record} likelihood={likelihood}"
            self.assertAlmostEqual(rz, rz_hat, places=12, msg=msg)
            self.assertAlmostEqual(vz, vz_hat, places=12, msg=msg)

    def _test_function_proba(self, likelihood, records):
        for record in records:
            az, tau = record["az"], record["tau"]

            def one(bz, y):
                return 1.
            sum_proba = likelihood.beliefs_measure(az, tau, f=one)
            msg = f"record={record} likelihood={likelihood}"
            self.assertAlmostEqual(sum_proba, 1., places=12, msg=msg)

    def test_sng_posterior(self):
        records = []
        likelihoods = [
            SngLikelihood(y = np.array([+1])),
            SngLikelihood(y = np.array([-1]))
        ]
        for likelihood in likelihoods:
            self._test_function_posterior(likelihood, records)

    def test_sng_proba(self):
        records = [
            dict(az=2.0, tau=1.3),
            dict(az=15., tau=1.0)
        ]
        likelihoods = [
            SngLikelihood(y = np.array([+1])),
            SngLikelihood(y = np.array([-1]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, records)

    def test_abs_proba(self):
        records = [
            dict(az=2.0, tau=1.3),
            dict(az=15., tau=1.0)
        ]
        likelihoods = [
            AbsLikelihood(y = np.array([10.4])),
            AbsLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, records)

    def test_modulus_proba(self):
        records = [
            dict(az=2.0, tau=1.3),
            dict(az=15., tau=1.0)
        ]
        likelihoods = [
            ModulusLikelihood(y = np.array([10.4])),
            ModulusLikelihood(y = np.array([1.3]))
        ]
        for likelihood in likelihoods:
            self._test_function_proba(likelihood, records)

if __name__ == "__main__":
    unittest.main()

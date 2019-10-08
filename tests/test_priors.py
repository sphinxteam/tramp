import unittest
from tramp.priors import BinaryPrior, GaussBernouilliPrior, GaussianPrior
import numpy as np


def empirical_second_moment(prior):
    """
    Estimate second_moment by sampling.
    """
    prior.size = 1000*1000
    X = prior.sample()
    tau_x = (X**2).mean()
    return tau_x


def explicit_integral(ax, bx, prior):
    """
    Compute rx, vx for prior p(x) by integration.
    """
    def belief(x):
        L = -0.5 * ax * (x**2) + bx * x
        return np.exp(L)

    def x_belief(x):
        return x*belief(x)

    def x2_belief(x):
        return (x**2)*belief(x)

    Z = prior.measure(belief)
    rx = prior.measure(x_belief) / Z
    x2 = prior.measure(x2_belief) / Z
    vx = x2 - rx**2
    return rx, vx


class PriorsTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            dict(ax=2.0, bx=2.0),
            dict(ax=1.5, bx=1.3)
        ]

    def tearDown(self):
        pass

    def _test_function_second_moment(self, prior, places=2):
        tau_x_emp = empirical_second_moment(prior)
        tau_x_hat = prior.second_moment()
        msg = f"prior={prior}"
        self.assertAlmostEqual(tau_x_emp, tau_x_hat, places=places, msg=msg)

    def _test_function_posterior(self, prior, records, places=12):
        for record in records:
            ax, bx = record["ax"], record["bx"]
            rx, vx = explicit_integral(ax, bx, prior)
            rx_hat, vx_hat = prior.compute_forward_posterior(ax, bx)
            rx_hat = float(rx_hat)
            msg = f"record={record} prior={prior}"
            self.assertAlmostEqual(rx, rx_hat, places=places, msg=msg)
            self.assertAlmostEqual(vx, vx_hat, places=places, msg=msg)

    def _test_function_proba(self, prior, records, places=12):
        for record in records:
            ax = record["ax"]
            def one(bx): return 1
            sum_proba = prior.beliefs_measure(ax, f=one)
            msg = f"record={record} prior={prior}"
            self.assertAlmostEqual(sum_proba, 1., places=places, msg=msg)

    def test_gaussian_posterior(self):
        priors = [
            GaussianPrior(size=1, mean=0.5, var=1.0),
            GaussianPrior(size=1, mean=-2.5, var=10.0)
        ]
        for prior in priors:
            self._test_function_posterior(prior, self.records)

    def test_binary_posterior(self):
        priors = [
            BinaryPrior(size=1, p_pos=0.5),
            BinaryPrior(size=1, p_pos=0.6)
        ]
        for prior in priors:
            self._test_function_posterior(prior, self.records)

    def test_gauss_bernouilli_posterior(self):
        priors = [
            GaussBernouilliPrior(size=1, rho=0.5, mean=0., var=0.8),
            GaussBernouilliPrior(size=1, rho=0.9, mean=1.5, var=1.0)
        ]
        for prior in priors:
            self._test_function_posterior(prior, self.records)

    def test_gaussian_second_moment(self):
        priors = [
            GaussianPrior(size=1, mean=0.5, var=1.0),
            GaussianPrior(size=1, mean=-0.3, var=0.5)
        ]
        for prior in priors:
            self._test_function_second_moment(prior)

    def test_binary_second_moment(self):
        priors = [
            BinaryPrior(size=1, p_pos=0.5),
            BinaryPrior(size=1, p_pos=0.6)
        ]
        for prior in priors:
            self._test_function_second_moment(prior)

    def test_gauss_bernouilli_second_moment(self):
        priors = [
            GaussBernouilliPrior(size=1, rho=0.5, mean=0., var=0.8),
            GaussBernouilliPrior(size=1, rho=0.9, mean=1.5, var=1.0)
        ]
        for prior in priors:
            self._test_function_second_moment(prior)

    def test_binary_proba(self):
        priors = [
            BinaryPrior(size=1, p_pos=0.5),
            BinaryPrior(size=1, p_pos=0.6)
        ]
        for prior in priors:
            self._test_function_proba(prior, self.records)

    def test_gauss_bernouilli_proba(self):
        priors = [
            GaussBernouilliPrior(size=1, rho=0.5, mean=0., var=0.8),
            GaussBernouilliPrior(size=1, rho=0.9, mean=1.5, var=1.0)
        ]
        for prior in priors:
            self._test_function_proba(prior, self.records)


if __name__ == "__main__":
    unittest.main()

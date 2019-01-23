import unittest
from tramp.priors import BinaryPrior, GaussBernouilliPrior, GaussianPrior
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import logging


def empirical_second_moment(prior):
    """
    Estimate second_moment by sampling.
    """
    prior.size = 1000*1000
    X = prior.sample()
    tau = (X**2).mean()
    return tau

def explicit_integral(ax, bx, prior):
    """
    Compute rx, vx for prior p(x) by integration
    """
    def belief(x):
        L = -0.5 * ax * (x**2) + bx * x
        return np.exp(L)

    if isinstance(prior, BinaryPrior):
        Z = prior.p_pos*belief(+1) + prior.p_neg*belief(-1)
        rx = (prior.p_pos*belief(+1) - prior.p_neg*belief(-1))/Z
        x2 = 1
        vx = x2 - rx**2
    if isinstance(prior, GaussianPrior):
        def integrand(x):
            return belief(x)*norm.pdf(x, loc=prior.mean, scale=prior.sigma)
        def x_integrand(x):
            return x*integrand(x)
        def x2_integrand(x):
            return (x**2)*integrand(x)
        xmin, xmax = -10, 10
        Z = quad(integrand, xmin, xmax)[0]
        rx = quad(x_integrand, xmin, xmax)[0] / Z
        x2 = quad(x2_integrand, xmin, xmax)[0] / Z
        vx = x2 - rx**2
    if isinstance(prior, GaussBernouilliPrior):
        def integrand(x):
            return belief(x)*norm.pdf(x, loc=prior.mean, scale=prior.sigma)
        def x_integrand(x):
            return x*integrand(x)
        def x2_integrand(x):
            return (x**2)*integrand(x)
        xmin, xmax = -10, 10
        Z = prior.rho * quad(integrand, xmin, xmax)[0] + (1-prior.rho)*belief(0)
        rx = prior.rho * quad(x_integrand, xmin, xmax)[0] / Z
        x2 = prior.rho * quad(x2_integrand, xmin, xmax)[0] / Z
        vx = x2 - rx**2

    return rx, vx

class PriorsTest(unittest.TestCase):
    def setUp(self):
        self.posterior_records = [
            dict(ax=2.0, bx=2.0),
            dict(ax=1.5, bx=1.3)
        ]

    def tearDown(self):
        pass

    def _test_function_second_moment(self, prior):
        tau = empirical_second_moment(prior)
        tau_hat = prior.second_moment()
        msg = f"prior={prior}"
        self.assertAlmostEqual(tau, tau_hat, places=2, msg=msg)

    def _test_function_posterior(self, prior):
        for record in self.posterior_records:
            ax, bx = record["ax"], record["bx"]
            rx, vx = explicit_integral(ax, bx, prior)
            rx_hat, vx_hat = prior.compute_forward_posterior(ax, bx)
            rx_hat = float(rx_hat)
            msg = f"record={record} prior={prior}"
            self.assertAlmostEqual(rx, rx_hat, places=12, msg=msg)
            self.assertAlmostEqual(vx, vx_hat, places=12, msg=msg)

    def _test_function_proba(self, prior):
        for record in self.posterior_records:
            ax = record["ax"]
            def one(bx):
                return 1.
            sum_proba = prior.beliefs_measure(ax, f = one)
            msg = f"record={record} prior={prior}"
            self.assertAlmostEqual(sum_proba, 1., places=12, msg=msg)

    def test_binary_posterior(self):
        priors = [
            BinaryPrior(size=1, p_pos=0.5),
            BinaryPrior(size=1, p_pos=0.6)
        ]
        for prior in priors:
            self._test_function_posterior(prior)

    def test_gauss_bernouilli_posterior(self):
        priors = [
            GaussBernouilliPrior(size=1, rho=0.5, mean=0., var=0.8),
            GaussBernouilliPrior(size=1, rho=0.9, mean=1.5, var=1.0)
        ]
        for prior in priors:
            self._test_function_posterior(prior)

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
            self._test_function_proba(prior)

    def test_gauss_bernouilli_proba(self):
        priors = [
            GaussBernouilliPrior(size=1, rho=0.5, mean=0., var=0.8),
            GaussBernouilliPrior(size=1, rho=0.9, mean=1.5, var=1.0)
        ]
        for prior in priors:
            self._test_function_proba(prior)

if __name__ == "__main__":
    unittest.main()

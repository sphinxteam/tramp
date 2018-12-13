import unittest
from tramp.channels import AbsChannel, SngChannel
import numpy as np
from scipy.integrate import quad, dblquad
import logging


def explicit_integral(az, bz, ax, bx, f):
    """
    Compute rx, vx, rz, vz for channel x = f(z) by explicit integration
    over the partition function
    """
    def integrand(z):
        x = f(z)
        L = -0.5 * ax * (x**2) + bx * x - 0.5 * az * (z**2) + bz * z
        return np.exp(L)

    def z_integrand(z):
        return z * integrand(z)

    def z2_integrand(z):
        return (z**2) * integrand(z)

    def x_integrand(z):
        return f(z) * integrand(z)

    def x2_integrand(z):
        return (f(z)**2) * integrand(z)

    zmin = -5
    zmax = 5
    Z = quad(integrand, zmin, zmax)[0]
    lnZ = np.log(Z)

    rx = quad(x_integrand, zmin, zmax)[0] / Z
    x2 = quad(x2_integrand, zmin, zmax)[0] / Z
    vx = x2 - rx**2

    rz = quad(z_integrand, zmin, zmax)[0] / Z
    z2 = quad(z2_integrand, zmin, zmax)[0] / Z
    vz = z2 - rz**2

    return rz, vz, rx, vx

class ChannelsTest(unittest.TestCase):
    def setUp(self):
        self.second_moment_records = [
            dict(mean=-3.2, sigma=1.7),
            dict(mean=0., sigma=2.0),
        ]
        self.posterior_records = [
            dict(az=2.0, bz=2.0, ax=2.0, bx=2.0, tau=1.3),
            dict(az=0.9, bz=1.6, ax=1.5, bx=1.3, tau=2.),
            dict(az=0.9, bz=-1.6, ax=1.5, bx=1.3, tau=2.)
        ]
        def parse_record_ab(record):
            return record["az"], record["bz"], record["ax"], record["bx"]
        def parse_record_ab_tau(record):
            return record["az"], record["bz"], record["ax"], record["bx"], record["tau"]
        self.parse_record_ab = parse_record_ab
        self.parse_record_ab_tau = parse_record_ab_tau

    def tearDown(self):
        pass

    def _test_function_second_moment(self, channel):
        for record in self.second_moment_records:
            noise = np.random.standard_normal(size=1000 * 1000)
            Z = record["mean"] + record["sigma"] * noise
            tau_Z = (Z**2).mean()
            X = channel.sample(Z)
            tau_X = (X**2).mean()
            tau_X_hat = channel.second_moment(tau_Z)
            msg = f"record={record}"
            self.assertAlmostEqual(tau_X_hat, tau_X, places=6, msg=msg)

    def _test_function_posterior(self, channel, f):
        # modify _parse_message_ab accept a record as input
        channel._parse_message_ab = self.parse_record_ab
        for record in self.posterior_records:
            az, bz, ax, bx = self.parse_record_ab(record)
            rz, vz, rx, vx = explicit_integral(az, bz, ax, bx, f)
            [(rx_hat, vx_hat)] = channel.forward_posterior(record)
            [(rz_hat, vz_hat)] = channel.backward_posterior(record)
            msg = f"record={record}"
            self.assertAlmostEqual(rx, rx_hat, places=1, msg=msg)
            self.assertAlmostEqual(vx, vx_hat, places=1, msg=msg)
            self.assertAlmostEqual(rz, rz_hat, places=1, msg=msg)
            self.assertAlmostEqual(vz, vz_hat, places=1 , msg=msg)

    def _test_function_proba(self, channel):
        # modify _parse_message_ab accept a record as input
        channel._parse_message_ab_tau = self.parse_record_ab_tau
        for record in self.posterior_records:
            az, _, ax, _, tau = self.parse_record_ab_tau(record)
            def proba_beliefs(x, z):
                _record = dict(bx=x, bz=z, az=az, ax=ax, tau=tau)
                return channel.proba_beliefs(_record)
            sum_proba = dblquad(proba_beliefs, -10, 10, -10, 10)[0]
            msg = f"record={record}"
            self.assertAlmostEqual(sum_proba, 1., places=6, msg=msg)

    def test_abs_posterior(self):
        channel = AbsChannel()
        self._test_function_posterior(channel, np.abs)

    def test_sng_posterior(self):
        channel = SngChannel()
        self._test_function_posterior(channel, np.sign)

    def test_abs_second_moment(self):
        channel = AbsChannel()
        self._test_function_second_moment(channel)

    def test_sng_second_moment(self):
        channel = SngChannel()
        self._test_function_second_moment(channel)

    def test_abs_proba(self):
        channel = AbsChannel()
        self._test_function_proba(channel)

    def test_sng_proba(self):
        channel = SngChannel()
        self._test_function_proba(channel)

if __name__ == "__main__":
    unittest.main()

import unittest
from tramp.channels import AbsChannel, SngChannel
import numpy as np


def empirical_second_moment(mean, sigma, channel):
    """
    Estimate second_moment by sampling.
    """
    noise = np.random.standard_normal(size=1000 * 1000)
    Z = mean + sigma * noise
    tau_Z = (Z**2).mean()
    X = channel.sample(Z)
    tau_X = (X**2).mean()
    return tau_Z, tau_X

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


def explicit_integral(az, bz, ax, bx, channel):
    """
    Compute rx, vx, rz, vz for p(x|z) by integration
    """
    def belief(z, x):
        L = -0.5 * ax * (x**2) + bx * x - 0.5 * az * (z**2) + bz * z
        return np.exp(L)
    def z_belief(z, x):
        return z * belief(z, x)
    def z2_belief(z, x):
        return (z**2) * belief(z, x)
    def x_belief(z, x):
        return x * belief(z, x)
    def x2_belief(z, x):
        return (x**2) * belief(z, x)

    zmin = bz / az - 10 / np.sqrt(az)
    zmax = bz / az + 10 / np.sqrt(az)

    Z = channel.measure(belief, zmin, zmax)
    rx = channel.measure(x_belief, zmin, zmax) / Z
    x2 = channel.measure(x2_belief, zmin, zmax) / Z
    vx = x2 - rx**2
    rz = channel.measure(z_belief, zmin, zmax) / Z
    z2 = channel.measure(z2_belief, zmin, zmax) / Z
    vz = z2 - rz**2

    return rz, vz, rx, vx

class ChannelsTest(unittest.TestCase):
    def setUp(self):
        self.second_moment_records = [
            dict(mean=-3.2, sigma=1.7),
            dict(mean=0., sigma=2.0),
        ]
        self.records = [
            dict(az=2.0, bz=2.0, ax=2.0, bx=2.0, tau=1.3),
            dict(az=0.9, bz=1.6, ax=1.5, bx=1.3, tau=2.),
            dict(az=0.9, bz=-1.6, ax=1.5, bx=1.3, tau=2.)
        ]

    def tearDown(self):
        pass

    def _test_function_second_moment(self, channel, records, places=6):
        for record in records:
            tau_Z, tau_X = empirical_second_moment(
                record["mean"], record["sigma"], channel
            )
            tau_X_hat = channel.second_moment(tau_Z)
            msg = f"record={record}"
            self.assertAlmostEqual(tau_X_hat, tau_X, places=places, msg=msg)

    def _test_function_posterior(self, channel, records, places=12):
        for record in records:
            az, bz, ax, bx = record["az"], record["bz"], record["ax"], record["bx"]
            rz, vz, rx, vx = explicit_integral(az, bz, ax, bx, channel)
            rx_hat, vx_hat = channel.compute_forward_posterior(az, bz, ax, bx)
            rz_hat, vz_hat = channel.compute_backward_posterior(az, bz, ax, bx)
            msg = f"record={record}"
            self.assertAlmostEqual(rx, rx_hat, places=places, msg=msg)
            self.assertAlmostEqual(vx, vx_hat, places=places, msg=msg)
            self.assertAlmostEqual(rz, rz_hat, places=places, msg=msg)
            self.assertAlmostEqual(vz, vz_hat, places=places, msg=msg)

    def _test_function_proba(self, channel, records, places=12):
        for record in records:
            az, ax, tau = record["az"], record["ax"], record["tau"]
            one = lambda bz, bx: 1
            sum_proba = channel.beliefs_measure(az, ax, tau, f=one)
            msg = f"record={record}"
            self.assertAlmostEqual(sum_proba, 1., places=places, msg=msg)

    def test_abs_posterior(self):
        channel = AbsChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_sng_posterior(self):
        channel = SngChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_abs_second_moment(self):
        channel = AbsChannel()
        self._test_function_second_moment(channel, self.second_moment_records)

    def test_sng_second_moment(self):
        channel = SngChannel()
        self._test_function_second_moment(channel, self.second_moment_records)

    def test_abs_proba(self):
        channel = AbsChannel()
        self._test_function_proba(channel, self.records)

    def test_sng_proba(self):
        channel = SngChannel()
        self._test_function_proba(channel, self.records)

if __name__ == "__main__":
    unittest.main()

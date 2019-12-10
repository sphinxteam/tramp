import unittest
from tramp.channels import (
    AbsChannel, SgnChannel, ReluChannel, LeakyReluChannel, HardTanhChannel
)
import numpy as np


def empirical_second_moment(tau_z, channel):
    """
    Estimate second_moment by sampling.
    """
    noise = np.random.standard_normal(size=1000 * 1000)
    Z = np.sqrt(tau_z) * noise / noise.std()
    X = channel.sample(Z)
    tau_x = (X**2).mean()
    return tau_x


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
        self.records = [
            dict(az=2.1, bz=2.0, ax=2.0, bx=2.0, tau_z=2.0),
            dict(az=2.0, bz=+1.6, ax=1.5, bx=1.3, tau_z=1.5),
            dict(az=2.0, bz=-1.6, ax=1.5, bx=1.3, tau_z=1.0)
        ]

    def tearDown(self):
        pass

    def _test_function_second_moment(self, channel, records, places=6):
        for record in records:
            tau_z = record["tau_z"]
            tau_x_emp = empirical_second_moment(tau_z, channel)
            tau_x_hat = channel.second_moment(tau_z)
            msg = f"record={record}"
            self.assertAlmostEqual(tau_x_emp, tau_x_hat, places=places, msg=msg)

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
            az, ax, tau_z = record["az"], record["ax"], record["tau_z"]
            def one(bz, bx): return 1
            sum_proba = channel.beliefs_measure(az, ax, tau_z, f=one)
            msg = f"record={record}"
            self.assertAlmostEqual(sum_proba, 1., places=places, msg=msg)

    def test_abs_posterior(self):
        channel = AbsChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_sgn_posterior(self):
        channel = SgnChannel()
        self._test_function_posterior(channel, self.records, places=4)

    def test_relu_posterior(self):
        channel = ReluChannel()
        self._test_function_posterior(channel, self.records, places=6)

    def test_leaky_relu_posterior(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_posterior(channel, self.records, places=6)

    def test_hard_tanh_posterior(self):
        channel = HardTanhChannel()
        self._test_function_posterior(channel, self.records, places=1)

    def test_abs_second_moment(self):
        channel = AbsChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_sgn_second_moment(self):
        channel = SgnChannel()
        self._test_function_second_moment(channel, self.records)

    def test_relu_second_moment(self):
        channel = ReluChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_leaky_relu_second_moment(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_second_moment(channel, self.records, places=2)

    def test_hard_tanh_second_moment(self):
        channel = HardTanhChannel()
        self._test_function_second_moment(channel, self.records, places=2)

    def test_abs_proba(self):
        channel = AbsChannel()
        self._test_function_proba(channel, self.records)

    def test_sgn_proba(self):
        channel = SgnChannel()
        self._test_function_proba(channel, self.records)

    def test_relu_proba(self):
        channel = ReluChannel()
        self._test_function_proba(channel, self.records)

    def test_leaky_relu_proba(self):
        channel = LeakyReluChannel(slope=0.1)
        self._test_function_proba(channel, self.records)


if __name__ == "__main__":
    unittest.main()

from ..base_channel import Channel


class ReshapeChannel(Channel):
    """
    Reshape array

    Parameters
    ----------
    - next_shape : output shape
    - prev_shape : input shape
    """

    def __init__(self, prev_shape, next_shape):
        self.prev_shape = prev_shape
        self.next_shape = next_shape
        self.repr_init()

    def sample(self, Z):
        return Z.reshape(self.next_shape)

    def math(self):
        return r"$\delta$"

    def second_moment(self, tau_z):
        return tau_z

    def compute_forward_message(self, az, bz, ax, bx):
        return az, bz.reshape(self.next_shape)

    def compute_backward_message(self, az, bz, ax, bx):
        return ax, bx.reshape(self.prev_shape)

    def compute_forward_state_evolution(self, az, ax, tau_z):
        return az

    def compute_backward_state_evolution(self, az, ax, tau_z):
        return ax

    def compute_log_partition(self, az, bz, ax, bx):
        a = az + ax
        b = bz + bx.rehape(self.prev_shape)
        logZ = 0.5 * np.sum(b**2 / a + np.log(2 * np.pi / a))
        return logZ

    def compute_mutual_information(self, az, ax, tau_z):
        a = ax + az
        I = 0.5*np.log(a*tau_z)
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

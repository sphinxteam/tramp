from ..base_channel import Channel


class ReshapeChannel(Channel):
    """
    Reshape vector into tensor of given shape.

    Notes
    -----
    - length of input vector must be consitent with the output shape.
    """

    def __init__(self, shape):
        self.shape = shape
        self.repr_init()

    def sample(self, Z):
        return Z.reshape(self.shape)

    def math(self):
        return r"$\delta$"

    def second_moment(self, tau_z):
        return tau_z

    def compute_forward_message(self, az, bz, ax, bx):
        return az, bz.reshape(self.shape)

    def compute_backward_message(self, az, bz, ax, bx):
        return ax, bx.ravel()

    def compute_forward_state_evolution(self, az, ax, tau_z):
        return az

    def compute_backward_state_evolution(self, az, ax, tau_z):
        return ax

    def log_partition(self, az, bz, ax, bx):
        a = az + ax
        b = bz + bx.rehape(self.shape)
        logZ = 0.5 * np.sum(b**2 / a + np.log(2 * np.pi / a))
        return logZ

    def mutual_information(self, az, ax, tau_z):
        a = ax + az
        I = 0.5*np.log(a*tau_z)
        return I

    def free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A

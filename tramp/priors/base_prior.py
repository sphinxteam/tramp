from ..base import Factor


class Prior(Factor):
    n_next = 1
    n_prev = 0

    def compute_forward_message(self, ax, bx):
        rx, vx = self.compute_forward_posterior(ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_forward_state_evolution(self, ax):
        vx = self.compute_forward_error(ax)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new

    def compute_forward_error(self, ax):
        def variance(bx):
            rx, vx = self.compute_forward_posterior(ax, bx)
            return vx
        error = self.beliefs_measure(ax, f=variance)
        return error

    def compute_forward_overlap(self, ax):
        vx = self.compute_forward_error(ax)
        tau_x = self.second_moment()
        mx = tau_x - vx
        return mx

    def compute_free_energy(self, ax):
        def log_partition(bx):
            return self.compute_log_partition(ax, bx)
        A = self.beliefs_measure(ax, f=log_partition)
        return A

    def compute_mutual_information(self, ax):
        tau_x = self.second_moment()
        A = self.compute_free_energy(ax)
        I = 0.5*ax*tau_x - A
        return A

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

    def free_energy(self, ax):
        def log_partition(bx):
            return self.log_partition(ax, bx)
        A = self.beliefs_measure(ax, f=log_partition)
        return A

"""Implements the base Prior class."""
from ..base import Factor
from scipy.optimize import root_scalar


class Prior(Factor):
    n_next = 1
    n_prev = 0

    def prior_log_partition_FG(self, tx_hat):
        return self.scalar_log_partition(ax=tx_hat, bx=0)

    def compute_forward_message(self, ax, bx):
        rx, vx = self.compute_forward_posterior(ax, bx)
        ax_new, bx_new = self.compute_ab_new(rx, vx, ax, bx)
        return ax_new, bx_new

    def compute_forward_state_evolution_BO(self, ax, tx0_hat):
        vx = self.compute_forward_v_BO(ax, tx0_hat)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new

    def compute_forward_v_BO(self, ax, tx0_hat):
        mx_hat = ax - tx0_hat
        def v_func(bx): return self.scalar_forward_variance(ax, bx)
        vx = self.b_measure(mx_hat, mx_hat, tx0_hat, v_func)
        return vx

    def compute_potential_BO(self, ax, tx0_hat):
        mx_hat = ax - tx0_hat
        def A_func(bx): return self.scalar_log_partition(ax, bx)
        A = self.b_measure(mx_hat, mx_hat, tx0_hat, A_func)
        return A

    def compute_forward_state_evolution_RS(self, ax, mx_hat, qx_hat,
                                           teacher, tx0_hat, tx0):
        vx, mx, qx = self.compute_forward_vmq_RS(
            ax, mx_hat, qx_hat, teacher, tx0_hat
        )
        ax_new, mx_hat_new, qx_hat_new = self.compute_a_mhat_qhat_new(
            vx, mx, qx, ax, mx_hat, qx_hat, tx0
        )
        return ax_new, mx_hat_new, qx_hat_new

    def compute_forward_vmq_RS(self, ax, mx_hat, qx_hat,
                               teacher, tx0_hat):
        def v_func(bx): return self.scalar_forward_variance(ax, bx)
        def r_func(bx): return self.scalar_forward_mean(ax, bx)
        def q_func(bx): return self.scalar_forward_mean(ax, bx)**2
        vx = teacher.b_measure(mx_hat, qx_hat, tx0_hat, v_func)
        mx = teacher.bx_measure(mx_hat, qx_hat, tx0_hat, r_func)
        qx = teacher.b_measure(mx_hat, qx_hat, tx0_hat, q_func)
        return vx, mx, qx

    def compute_potential_RS(self, ax, mx_hat, qx_hat,
                             teacher, tx0_hat):
        def A_func(bx): return self.scalar_log_partition(ax, bx)
        A = teacher.b_measure(mx_hat, qx_hat, tx0_hat, A_func)
        return A

    def compute_forward_state_evolution(self, ax):
        vx = self.compute_forward_error(ax)
        ax_new = self.compute_a_new(vx, ax)
        return ax_new

    def compute_forward_error(self, ax):
        def v_func(bx): return self.scalar_forward_variance(ax, bx)
        error = self.beliefs_measure(ax, v_func)
        return error

    def compute_forward_overlap(self, ax):
        vx = self.compute_forward_error(ax)
        tau_x = self.second_moment()
        mx = tau_x - vx
        return mx

    def compute_free_energy(self, ax):
        def A_func(bx): return self.scalar_log_partition(ax, bx)
        A = self.beliefs_measure(ax, A_func)
        return A

    def compute_mutual_information(self, ax):
        tau_x = self.second_moment()
        A = self.compute_free_energy(ax)
        I = 0.5*ax*tau_x - A
        return I

    def compute_precision(self, vx):
        def f(ax):
            return self.compute_forward_error(ax) - vx
        sol = root_scalar(f, bracket=[0, 1/vx], method='bisect')
        ax = sol.root
        return ax

    def compute_dual_mutual_information(self, vx):
        ax = self.compute_precision(vx)
        I = self.compute_mutual_information(ax)
        I_dual = I - 0.5*ax*vx
        return I_dual

    def compute_dual_free_energy(self, mx):
        tau_x = self.second_moment()
        vx = tau_x - mx
        ax = self.compute_precision(vx)
        A = self.compute_free_energy(ax)
        A_dual = 0.5*ax*mx - A
        return A_dual

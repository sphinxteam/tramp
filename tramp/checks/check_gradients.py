import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..experiments import simple_run_experiments
from ..beliefs import exponential, mixture


def numerical_1st_derivative(x, f, epsilon):
    "Compute numerical first derivative for f : R -> R"
    f1 = (f(x+0.5*epsilon) - f(x-0.5*epsilon)) / epsilon
    return f1


def numerical_2nd_derivative(x, f, epsilon):
    "Compute numerical second derivative for f : R -> R"
    f2 = (f(x+epsilon) - 2*f(x) + f(x-epsilon)) / (epsilon**2)
    return f2


def numerical_gradient(x, f, epsilon):
    "Compute numerical gradient for f : R^N -> R and x in R^N"
    grad_f = np.zeros(x.shape)
    indicator = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        indicator[idx] = 1
        i = 0.5*epsilon*indicator
        grad_f[idx] = (f(x+i)-f(x-i)) / epsilon
        indicator[idx] = 0
    return grad_f


def numerical_hessian_diagonal(x, f, epsilon):
    "Compute numerical Hessian diagonal for f : R^N -> R and x in R^N"
    grad2_f = np.zeros(x.shape)
    indicator = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        indicator[idx] = 1
        i = epsilon*indicator
        grad2_f[idx] = (f(x+i) - 2*f(x) + f(x-i)) / (epsilon**2)
        indicator[idx] = 0
    return grad2_f


EPSILON = 1e-3


def get_mixture_grad_b(b, a, b0, eta):
    "Assumes b scalar but a, b0, eta are K-arrays"
    def A_func(b):
        return mixture.A(a=a, b=b+b0, eta=eta)
    A1 = numerical_1st_derivative(b, A_func, EPSILON)
    A2 = numerical_2nd_derivative(b, A_func, EPSILON)
    r = mixture.r(a=a, b=b+b0, eta=eta)
    v = mixture.v(a=a, b=b+b0, eta=eta)
    return dict(b=b, r=r, v=v, A1=A1, A2=A2)


def check_mixture_grad_b(b_values, a, b0, eta):
    assert (
        type(a) == type(b0) == type(eta) == np.ndarray and a.shape == b0.shape == eta.shape
    ), "Must provide K-arrays a, b0, eta for mixture belief"
    df = pd.DataFrame([
        get_mixture_grad_b(b, a, b0, eta) for b in b_values
    ])
    return df


def get_belief_grad_b(belief, b, **kwargs):
    def A_func(b): return belief.A(b=b, **kwargs)
    A1 = numerical_1st_derivative(b, A_func, EPSILON)
    A2 = numerical_2nd_derivative(b, A_func, EPSILON)
    r = belief.r(b=b, **kwargs)
    v = belief.v(b=b, **kwargs)
    return dict(r=r, v=v, A1=A1, A2=A2)


def check_belief_grad_b(belief, **kwargs):
    b_values = np.linspace(-6, 6, 100)
    # mixture special case : b broadcasted to K-array
    if belief == mixture:
        return check_mixture_grad_b(b_values, **kwargs)
    # exponential special case : b must be negative
    if belief == exponential:
        b_values = np.linspace(-6, -1, 100)
    df = simple_run_experiments(
        get_belief_grad_b, belief=belief, b=b_values, **kwargs
    )
    return df


def plot_belief_grad_b(belief, **kwargs):
    df = check_belief_grad_b(belief, **kwargs)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(df["b"], df["r"], '-', label="r")
    axs[0].plot(df["b"], df["A1"], '--', label=r"$\partial_{b} A$")
    axs[0].set(xlabel="b")
    axs[0].legend()
    axs[1].plot(df["b"], df["v"], '-', label="v")
    axs[1].plot(df["b"], df["A2"], '--', label=r"$\partial^2_{b} A$")
    axs[1].set(xlabel="b")
    # adjust axs[1] limits when v is almost constant
    ylim = axs[1].get_ylim()
    if (ylim[1]-ylim[0])<EPSILON:
        axs[1].set_ylim(ylim[0]-0.12, ylim[1]+0.12)
    axs[1].legend()
    kwargs_str = " ".join(f"{key}={val}" for key, val in kwargs.items())
    fig.suptitle(f"{belief.__name__} {kwargs_str}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_prior_grad_RS(teacher, student, mx_hat, qx_hat, tx_hat, tx0_hat):
    def A_RS(mx_hat, qx_hat, tx_hat):
        ax = qx_hat + tx_hat
        return student.compute_potential_RS(ax, mx_hat, qx_hat, teacher, tx0_hat)
    # gradients
    def A_func(mx_hat): return A_RS(mx_hat, qx_hat, tx_hat)
    grad_mx_hat_A = numerical_1st_derivative(mx_hat, A_func, EPSILON)
    def A_func(qx_hat): return A_RS(mx_hat, qx_hat, tx_hat)
    grad_qx_hat_A = numerical_1st_derivative(qx_hat, A_func, EPSILON)
    def A_func(tx_hat): return A_RS(mx_hat, qx_hat, tx_hat)
    grad_tx_hat_A = numerical_1st_derivative(tx_hat, A_func, EPSILON)
    # m, q, tau
    ax = qx_hat + tx_hat
    vx, mx, qx = student.compute_forward_vmq_RS(ax, mx_hat, qx_hat, teacher, tx0_hat)
    tx = qx + vx
    return {
        "grad_mx_hat_A": grad_mx_hat_A,
        "grad_qx_hat_A": grad_qx_hat_A,
        "grad_tx_hat_A": grad_tx_hat_A,
        "mx": mx, "qx": qx, "vx": vx, "tx": tx
    }


def check_prior_grad_RS(teacher, student):
    df = simple_run_experiments(
        get_prior_grad_RS, teacher=teacher, student=student,
        mx_hat=np.linspace(1, 3, 10), qx_hat=1, tx_hat=1, tx0_hat=1
    )
    return df


def plot_prior_grad_RS(teacher, student):
    df = check_prior_grad_RS(teacher, student)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(df["mx_hat"], df["mx"], '-', label=r"$m_x$")
    axs[0].plot(df["mx_hat"], df["grad_mx_hat_A"], '--', label=r"$\partial_{\widehat{m}_x^-} A$")
    axs[0].set(xlabel=r"$\widehat{m}_x^-$")
    axs[0].legend()
    axs[1].plot(df["mx_hat"], df["qx"], '-', label=r"$q_x$")
    axs[1].plot(df["mx_hat"], -2*df["grad_qx_hat_A"], '--', label=r"$-2\partial_{\widehat{q}_x^-} A$")
    axs[1].set(xlabel=r"$\widehat{m}_x^-$")
    axs[1].legend()
    axs[2].plot(df["mx_hat"], df["tx"], '-', label=r"$\tau_x$")
    axs[2].plot(df["mx_hat"], -2*df["grad_tx_hat_A"], '--', label=r"$-2\partial_{\widehat{\tau}_x^-} A$")
    axs[2].set(xlabel=r"$\widehat{m}_x^-$")
    axs[2].legend()
    fig.suptitle(f"teacher={teacher}\nstudent={student}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_prior_grad_BO(prior, mx_hat, tx0_hat):
    def A_func(mx_hat):
        ax = mx_hat + tx0_hat
        return prior.compute_potential_BO(ax=ax, tx0_hat=tx0_hat)
    grad_mx_hat_A = numerical_1st_derivative(mx_hat, A_func, EPSILON)
    ax = mx_hat + tx0_hat
    vx = prior.compute_forward_v_BO(ax=ax, tx0_hat=tx0_hat)
    tx = prior.second_moment_FG(tx_hat=tx0_hat)
    mx = tx - vx
    return {
        "grad_mx_hat_A": grad_mx_hat_A,
        "mx": mx, "tx": tx, "vx": vx
    }


def check_prior_grad_BO(prior):
    df = simple_run_experiments(
        get_prior_grad_BO, prior=prior, mx_hat=np.linspace(1, 3, 10), tx0_hat=1
    )
    return df


def plot_prior_grad_BO(prior):
    df = check_prior_grad_BO(prior)
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.plot(df["mx_hat"], df["mx"], '-', label=r"$m_x$")
    axs.plot(df["mx_hat"], 2*df["grad_mx_hat_A"], '--', label=r"$2\partial_{\widehat{m}_x^-} A$")
    axs.set(xlabel=r"$\widehat{m}_x^-$")
    axs.legend()
    fig.suptitle(f"{prior}".replace("(", "\n").replace(")", "\n"), fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_prior_grad_BO_BN(prior, ax):
    A_func = prior.compute_free_energy
    I_func = prior.compute_mutual_information
    return {
        "grad_ax_A": numerical_1st_derivative(ax, A_func, EPSILON),
        "grad_ax_I": numerical_1st_derivative(ax, I_func, EPSILON),
        "mx": prior.compute_forward_overlap(ax),
        "vx": prior.compute_forward_error(ax),
    }


def check_prior_grad_BO_BN(prior):
    df = simple_run_experiments(
        get_prior_grad_BO_BN, prior=prior, ax=np.linspace(1, 3, 10)
    )
    return df


def plot_prior_grad_BO_BN(prior):
    df = check_prior_grad_BO_BN(prior)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(df["ax"], df["mx"], '-', label=r"$m_x$")
    axs[0].plot(df["ax"], 2*df["grad_ax_A"], '--', label=r"$2\partial_{a_x^-} A$")
    axs[0].set(xlabel=r"$a_x^-$")
    axs[0].legend()
    axs[1].plot(df["ax"], df["vx"], '-', label=r"$v_x$")
    axs[1].plot(df["ax"], 2*df["grad_ax_I"], '--', label=r"$2\partial_{a_x^-} I$")
    axs[1].set(xlabel=r"$a_x^-$")
    axs[1].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_prior_grad_FG(prior, tx_hat):
    def A_func(tx_hat): return prior.prior_log_partition_FG(tx_hat)
    grad_tx_hat_A = numerical_1st_derivative(tx_hat, A_func, EPSILON)
    tx = prior.second_moment_FG(tx_hat)
    return {"grad_tx_hat_A": grad_tx_hat_A, "tx": tx}


def check_prior_grad_FG(prior):
    df = simple_run_experiments(
        get_prior_grad_FG, prior=prior, tx_hat=np.linspace(1, 3, 10)
    )
    return df


def plot_prior_grad_FG(prior):
    df = check_prior_grad_FG(prior)
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.plot(df["tx_hat"], df["tx"], '-', label=r"$\tau_x$")
    axs.plot(df["tx_hat"], -2*df["grad_tx_hat_A"], '--', label=r"$-2\partial_{\widehat{\tau}_x^-} A$")
    axs.set(xlabel=r"$\widehat{\tau}_x^-$")
    axs.legend()
    fig.suptitle(f"{prior}".replace("(", "\n").replace(")", "\n"), fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_prior_grad_EP_scalar(prior, ax, bx):
    def A_func(bx): return prior.scalar_log_partition(ax, bx)
    grad_bx_A1 = numerical_1st_derivative(bx, A_func, EPSILON)
    grad_bx_A2 = numerical_2nd_derivative(bx, A_func, EPSILON)
    rx = prior.scalar_forward_mean(ax, bx)
    vx = prior.scalar_forward_variance(ax, bx)
    def A_func(ax): return prior.scalar_log_partition(ax, bx)
    grad_ax_A = numerical_1st_derivative(ax, A_func, EPSILON)
    qx = rx**2
    tx = qx + vx
    return {
        "grad_bx_A1": grad_bx_A1, "grad_bx_A2": grad_bx_A2, "grad_ax_A": grad_ax_A,
        "rx": rx, "vx": vx, "tx": tx, "qx": qx
    }


def check_prior_grad_EP_scalar(prior):
    df = simple_run_experiments(
        get_prior_grad_EP_scalar, prior=prior, ax=1., bx=np.linspace(-6, 6, 30)
    )
    return df


def plot_prior_grad_EP_scalar(prior):
    df = check_prior_grad_EP_scalar(prior)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(df["bx"], df["rx"], '-', label=r"$r_x$")
    axs[0].plot(df["bx"], df["grad_bx_A1"], '--', label=r"$\partial_{b_x^-} A$")
    axs[0].set(xlabel=r"$b_x^-$")
    axs[0].legend()
    axs[1].plot(df["bx"], df["vx"], '-', label=r"$v_x$")
    axs[1].plot(df["bx"], df["grad_bx_A2"], '--', label=r"$\partial^2_{b_x^-} A$")
    axs[1].set(xlabel=r"$b_x^-$")
    axs[1].legend()
    if prior.__class__.__name__.startswith("MAP"):
        axs[2].plot(df["bx"], df["qx"], '-', label=r"$q_x$")
    else:
        axs[2].plot(df["bx"], df["tx"], '-', label=r"$\tau_x$")
    axs[2].plot(df["bx"], -2*df["grad_ax_A"], '--', label=r"$-2\partial_{a_x^-} A$")
    axs[2].set(xlabel=r"$b_x^-$")
    axs[2].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def check_prior_grad_EP_diagonal(prior):
    assert not prior.isotropic, "Must use diagonal beliefs (isotropic=False)"
    N = np.prod(prior.size)
    bx = np.linspace(-6, 6, N).reshape(prior.size)
    ax = np.ones_like(bx)
    def A_func(bx):
        return N*prior.compute_log_partition(ax, bx)
    A1 = numerical_gradient(bx, A_func, EPSILON)
    A2 = numerical_hessian_diagonal(bx, A_func, EPSILON)
    rx, vx = prior.compute_forward_posterior(ax, bx)
    df = pd.DataFrame({
        "bx": bx.ravel(), "rx": rx.ravel(), "vx":vx.ravel(),
        "grad_bx_A1":A1.ravel(), "grad_bx_A2":A2.ravel()
    })
    return df


def plot_prior_grad_EP_diagonal(prior):
    df = check_prior_grad_EP_diagonal(prior)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(df["bx"], df["rx"], '-', label=r"$r_x$")
    axs[0].plot(df["bx"], df["grad_bx_A1"], '--', label=r"$\partial_{b_x^-} A$")
    axs[0].set(xlabel=r"$b_x^-$")
    axs[0].legend()
    axs[1].plot(df["bx"], df["vx"], '-', label=r"$v_x$")
    axs[1].plot(df["bx"], df["grad_bx_A2"], '--', label=r"$\partial^2_{b_x^-} A$")
    axs[1].set(xlabel=r"$b_x^-$")
    axs[1].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

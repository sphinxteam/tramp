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


"""GRADIENT CHECK BELIEF"""

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


"""GRADIENT CHECK PRIOR"""

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
    tx = prior.forward_second_moment_FG(tx_hat=tx0_hat)
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
    fig.suptitle(f"{prior}".replace("(", "\n").replace(")", "\n"), fontsize=11)
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
    tx = prior.forward_second_moment_FG(tx_hat)
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
    fig.suptitle(f"{prior}".replace("(", "\n").replace(")", "\n"), fontsize=11)
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


"""GRADIENT CHECK LIKELIHOOD"""

def get_likelihood_grad_RS(teacher, student, mz_hat, qz_hat, tz_hat, tz0_hat):
    def A_RS(mz_hat, qz_hat, tz_hat):
        az = qz_hat + tz_hat
        return student.compute_potential_RS(az, mz_hat, qz_hat, teacher, tz0_hat)
    # gradients
    def A_func(mz_hat): return A_RS(mz_hat, qz_hat, tz_hat)
    grad_mz_hat_A = numerical_1st_derivative(mz_hat, A_func, EPSILON)
    def A_func(qz_hat): return A_RS(mz_hat, qz_hat, tz_hat)
    grad_qz_hat_A = numerical_1st_derivative(qz_hat, A_func, EPSILON)
    def A_func(tz_hat): return A_RS(mz_hat, qz_hat, tz_hat)
    grad_tz_hat_A = numerical_1st_derivative(tz_hat, A_func, EPSILON)
    # m, q, tau
    az = qz_hat + tz_hat
    vz, mz, qz = student.compute_backward_vmq_RS(az, mz_hat, qz_hat, teacher, tz0_hat)
    tz = qz + vz
    return {
        "grad_mz_hat_A": grad_mz_hat_A,
        "grad_qz_hat_A": grad_qz_hat_A,
        "grad_tz_hat_A": grad_tz_hat_A,
        "mz": mz, "qz": qz, "vz": vz, "tz": tz
    }


def check_likelihood_grad_RS(teacher, student):
    df = simple_run_experiments(
        get_likelihood_grad_RS, teacher=teacher, student=student,
        mz_hat=np.linspace(1, 3, 10), qz_hat=1, tz_hat=1, tz0_hat=1
    )
    return df


def plot_likelihood_grad_RS(teacher, student):
    df = check_likelihood_grad_RS(teacher, student)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(df["mz_hat"], df["mz"], '-', label=r"$m_z$")
    axs[0].plot(df["mz_hat"], df["grad_mz_hat_A"], '--', label=r"$\partial_{\widehat{m}_z^+} A$")
    axs[0].set(xlabel=r"$\widehat{m}_z^+$")
    axs[0].legend()
    axs[1].plot(df["mz_hat"], df["qz"], '-', label=r"$q_z$")
    axs[1].plot(df["mz_hat"], -2*df["grad_qz_hat_A"], '--', label=r"$-2\partial_{\widehat{q}_z^+} A$")
    axs[1].set(xlabel=r"$\widehat{m}_z^+$")
    axs[1].legend()
    axs[2].plot(df["mz_hat"], df["tz"], '-', label=r"$\tau_z$")
    axs[2].plot(df["mz_hat"], -2*df["grad_tz_hat_A"], '--', label=r"$-2\partial_{\widehat{\tau}_z^+} A$")
    axs[2].set(xlabel=r"$\widehat{m}_z^+$")
    axs[2].legend()
    fig.suptitle(f"teacher={teacher}\nstudent={student}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_likelihood_grad_BO(likelihood, mz_hat, tz0_hat):
    def A_func(mz_hat):
        az = mz_hat + tz0_hat
        return likelihood.compute_potential_BO(az=az, tz0_hat=tz0_hat)
    grad_mz_hat_A = numerical_1st_derivative(mz_hat, A_func, EPSILON)
    az = mz_hat + tz0_hat
    vz = likelihood.compute_backward_v_BO(az=az, tz0_hat=tz0_hat)
    tz = likelihood.backward_second_moment_FG(tz_hat=tz0_hat)
    mz = tz - vz
    return {
        "grad_mz_hat_A": grad_mz_hat_A,
        "mz": mz, "tz": tz, "vz": vz
    }


def check_likelihood_grad_BO(likelihood):
    df = simple_run_experiments(
        get_likelihood_grad_BO, likelihood=likelihood,
        mz_hat=np.linspace(1, 3, 10), tz0_hat=1
    )
    return df


def plot_likelihood_grad_BO(likelihood):
    df = check_likelihood_grad_BO(likelihood)
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.plot(df["mz_hat"], df["mz"], '-', label=r"$m_z$")
    axs.plot(df["mz_hat"], 2*df["grad_mz_hat_A"], '--', label=r"$2\partial_{\widehat{m}_z^+} A$")
    axs.set(xlabel=r"$\widehat{m}_z^+$")
    axs.legend()
    fig.suptitle(f"{likelihood}".replace("(", "\n").replace(")", "\n"), fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_likelihood_grad_BO_BN(likelihood, az, tau_z):
    def A_func(az): return likelihood.compute_free_energy(az, tau_z)
    def I_func(az): return likelihood.compute_mutual_information(az, tau_z)
    return {
        "grad_az_A": numerical_1st_derivative(az, A_func, EPSILON),
        "grad_az_I": numerical_1st_derivative(az, I_func, EPSILON),
        "mz": likelihood.compute_backward_overlap(az, tau_z),
        "vz": likelihood.compute_backward_error(az, tau_z),
    }


def check_likelihood_grad_BO_BN(likelihood):
    df = simple_run_experiments(
        get_likelihood_grad_BO_BN, likelihood=likelihood,
        az=np.linspace(1, 3, 10), tau_z=2
    )
    return df


def plot_likelihood_grad_BO_BN(likelihood):
    df = check_likelihood_grad_BO_BN(likelihood)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(df["az"], df["mz"], '-', label=r"$m_z$")
    axs[0].plot(df["az"], 2*df["grad_az_A"], '--', label=r"$2\partial_{a_z^+} A$")
    axs[0].set(xlabel=r"$a_z^+$")
    axs[0].legend()
    axs[1].plot(df["az"], df["vz"], '-', label=r"$v_z$")
    axs[1].plot(df["az"], 2*df["grad_az_I"], '--', label=r"$2\partial_{a_z^+} I$")
    axs[1].set(xlabel=r"$a_z^+$")
    axs[1].legend()
    fig.suptitle(likelihood)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_likelihood_grad_FG(likelihood, tz_hat):
    def A_func(tz_hat): return likelihood.prior_log_partition_FG(tz_hat)
    grad_tz_hat_A = numerical_1st_derivative(tz_hat, A_func, EPSILON)
    tz = likelihood.backward_second_moment_FG(tz_hat)
    return {"grad_tz_hat_A": grad_tz_hat_A, "tz": tz}


def check_likelihood_grad_FG(likelihood):
    df = simple_run_experiments(
        get_likelihood_grad_FG, likelihood=likelihood, tz_hat=np.linspace(1, 3, 10)
    )
    return df


def plot_likelihood_grad_FG(likelihood):
    df = check_likelihood_grad_FG(likelihood)
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    axs.plot(df["tz_hat"], df["tz"], '-', label=r"$\tau_z$")
    axs.plot(df["tz_hat"], -2*df["grad_tz_hat_A"], '--', label=r"$-2\partial_{\widehat{\tau}_z^+} A$")
    axs.set(xlabel=r"$\widehat{\tau}_z^+$")
    axs.legend()
    fig.suptitle(f"{likelihood}".replace("(", "\n").replace(")", "\n"), fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


def get_likelihood_grad_EP_scalar(likelihood, az, bz):
    y = likelihood.y[0]
    def A_func(bz): return likelihood.scalar_log_partition(az, bz, y)
    grad_bz_A1 = numerical_1st_derivative(bz, A_func, EPSILON)
    grad_bz_A2 = numerical_2nd_derivative(bz, A_func, EPSILON)
    rz = likelihood.scalar_backward_mean(az, bz, y)
    vz = likelihood.scalar_backward_variance(az, bz, y)
    def A_func(az): return likelihood.scalar_log_partition(az, bz, y)
    grad_az_A = numerical_1st_derivative(az, A_func, EPSILON)
    qz = rz**2
    tz = qz + vz
    return {
        "grad_bz_A1": grad_bz_A1, "grad_bz_A2": grad_bz_A2, "grad_az_A": grad_az_A,
        "rz": rz, "vz": vz, "tz": tz, "qz": qz, "y":y
    }


def check_likelihood_grad_EP_scalar(likelihood):
    df = simple_run_experiments(
        get_likelihood_grad_EP_scalar, likelihood=likelihood, az=1., bz=np.linspace(-6, 6, 30)
    )
    return df


def plot_likelihood_grad_EP_scalar(likelihood):
    df = check_likelihood_grad_EP_scalar(likelihood)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(df["bz"], df["rz"], '-', label=r"$r_z$")
    axs[0].plot(df["bz"], df["grad_bz_A1"], '--', label=r"$\partial_{b_z^+} A$")
    axs[0].set(xlabel=r"$b_z^+$")
    axs[0].legend()
    axs[1].plot(df["bz"], df["vz"], '-', label=r"$v_z$")
    axs[1].plot(df["bz"], df["grad_bz_A2"], '--', label=r"$\partial^2_{b_z^+} A$")
    axs[1].set(xlabel=r"$b_z^+$")
    axs[1].legend()
    axs[2].plot(df["bz"], df["tz"], '-', label=r"$\tau_z$")
    axs[2].plot(df["bz"], -2*df["grad_az_A"], '--', label=r"$-2\partial_{a_z^+} A$")
    axs[2].set(xlabel=r"$b_z^+$")
    axs[2].legend()
    fig.suptitle(likelihood)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def check_likelihood_grad_EP_diagonal(likelihood):
    assert not likelihood.isotropic, "Must use diagonal beliefs (isotropic=False)"
    N = np.prod(likelihood.size)
    bz = np.linspace(-6, 6, N).reshape(likelihood.size)
    az = np.ones_like(bz)
    def A_func(bz):
        return N*likelihood.compute_log_partition(az, bz, likelihood.y)
    A1 = numerical_gradient(bz, A_func, EPSILON)
    A2 = numerical_hessian_diagonal(bz, A_func, EPSILON)
    rz, vz = likelihood.compute_backward_posterior(az, bz, likelihood.y)
    df = pd.DataFrame({
        "bz": bz.ravel(), "rz": rz.ravel(), "vz":vz.ravel(),
        "grad_bz_A1":A1.ravel(), "grad_bz_A2":A2.ravel()
    })
    return df


def plot_likelihood_grad_EP_diagonal(likelihood):
    df = check_likelihood_grad_EP_diagonal(likelihood)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(df["bz"], df["rz"], '-', label=r"$r_z$")
    axs[0].plot(df["bz"], df["grad_bz_A1"], '--', label=r"$\partial_{b_z^+} A$")
    axs[0].set(xlabel=r"$b_z^+$")
    axs[0].legend()
    axs[1].plot(df["bz"], df["vz"], '-', label=r"$v_z$")
    axs[1].plot(df["bz"], df["grad_bz_A2"], '--', label=r"$\partial^2_{b_z^+} A$")
    axs[1].set(xlabel=r"$b_z^+$")
    axs[1].legend()
    fig.suptitle(likelihood)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

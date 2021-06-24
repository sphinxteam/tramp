import numpy as np
import matplotlib.pyplot as plt
from ..experiments import simple_run_experiments


def get_prior_BO_limit(prior, mx_hat, tx0_hat):
    ax = mx_hat + tx0_hat
    A_BO  = prior.compute_potential_BO(ax=ax, tx0_hat=tx0_hat)
    vx_BO = prior.compute_forward_v_BO(ax=ax, tx0_hat=tx0_hat)
    tau_x = prior.second_moment_FG(tx_hat=tx0_hat)
    mx_BO = tau_x - vx_BO
    A_RS = prior.compute_potential_RS(
        ax=ax, mx_hat=mx_hat, qx_hat=mx_hat, teacher=prior, tx0_hat=tx0_hat
    )
    vx_RS, mx_RS, qx_RS = prior.compute_forward_vmq_RS(
        ax=ax, mx_hat=mx_hat, qx_hat=mx_hat, teacher=prior, tx0_hat=tx0_hat
    )
    return {
        "A_BO":A_BO, "vx_BO":vx_BO, "mx_BO":mx_BO,
        "A_RS":A_RS, "vx_RS":vx_RS, "mx_RS":mx_RS, "qx_RS":qx_RS
    }

def check_prior_BO_limit(prior):
    df = simple_run_experiments(
        get_prior_BO_limit, prior=prior,
        mx_hat=np.linspace(1, 3, 10), tx0_hat=1.
    )
    return df


def plot_prior_BO_limit(prior):
    df = check_prior_BO_limit(prior)
    fig, axs = plt.subplots(1, 3, figsize=(12,4), sharex=True)
    axs[0].plot(df["mx_hat"], df["A_BO"], "-", label=r"$A \quad BO$")
    axs[0].plot(df["mx_hat"], df["A_RS"], "--", label=r"$A \quad RS$")
    axs[0].set(xlabel=r"$\widehat{m}_x^-$")
    axs[0].legend()
    axs[1].plot(df["mx_hat"], df["vx_BO"], "-", label=r"$v_x \quad BO$")
    axs[1].plot(df["mx_hat"], df["vx_RS"], "--", label=r"$v_x \quad RS$")
    axs[1].set(xlabel=r"$\widehat{m}_x^-$")
    axs[1].legend()
    axs[2].plot(df["mx_hat"], df["mx_BO"], "-", label=r"$m_x \quad BO$")
    axs[2].plot(df["mx_hat"], df["mx_RS"], "--", label=r"$m_x \quad RS$")
    axs[2].plot(df["mx_hat"], df["qx_RS"], "x", label=r"$q_x \quad RS$")
    axs[2].set(xlabel=r"$\widehat{m}_x^-$")
    axs[2].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_prior_BN_limit(prior, mx_hat):
    ax = mx_hat
    A_FG  = prior.compute_potential_BO(ax=ax, tx0_hat=0)
    vx_FG = prior.compute_forward_v_BO(ax=ax, tx0_hat=0)
    tx_FG = prior.second_moment_FG(tx_hat=0)
    A_BN = prior.compute_free_energy(ax=ax)
    vx_BN = prior.compute_forward_error(ax=ax)
    tx_BN = prior.second_moment()
    return {
        "A_FG":A_FG, "vx_FG":vx_FG, "tx_FG":tx_FG,
        "A_BN":A_BN, "vx_BN":vx_BN, "tx_BN":tx_BN
    }


def check_prior_BN_limit(prior):
    df = simple_run_experiments(
        get_prior_BN_limit, prior=prior, mx_hat=np.linspace(1, 3, 10)
    )
    return df


def plot_prior_BN_limit(prior):
    df = check_prior_BN_limit(prior)
    fig, axs = plt.subplots(1, 3, figsize=(12,4), sharex=True)
    axs[0].plot(df["mx_hat"], df["A_BN"], "-", label=r"$A \quad BN$")
    axs[0].plot(df["mx_hat"], df["A_FG"], "--", label=r"$A \quad FG$")
    axs[0].set(xlabel=r"$\widehat{m}_x^-$")
    axs[0].legend()
    axs[1].plot(df["mx_hat"], df["vx_BN"], "-", label=r"$v_x \quad BN$")
    axs[1].plot(df["mx_hat"], df["vx_FG"], "--", label=r"$vx \quad FG$")
    axs[1].set(xlabel=r"$\widehat{m}_x^-$")
    axs[1].legend()
    axs[2].plot(df["mx_hat"], df["tx_BN"], "-", label=r"$\tau_x \quad BN$")
    axs[2].plot(df["mx_hat"], df["tx_FG"], "--", label=r"$\tau_x \quad FG$")
    axs[2].set(xlabel=r"$\widehat{m}_x^-$")
    axs[2].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

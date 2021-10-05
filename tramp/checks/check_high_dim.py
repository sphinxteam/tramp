import numpy as np
import matplotlib.pyplot as plt
from ..experiments import simple_run_experiments


"""CHECK HIGH DIM PRIOR"""


def get_prior_BO_BN_instance(prior, ax, sample):
    # sample
    x_true = prior.sample()
    noise = np.random.standard_normal(prior.size)
    bx = ax*x_true + np.sqrt(ax)*noise
    # single instance
    rx, vx = prior.compute_forward_posterior(ax, bx)
    vx = np.mean(vx)
    mx = np.mean(x_true*rx)
    qx = np.mean(rx**2)
    mse_x = np.mean((x_true-rx)**2)
    A = prior.compute_log_partition(ax, bx)
    return dict(vx=vx, mx=mx, qx=qx, mse_x=mse_x, A=A)


def get_prior_BO_BN_ensemble(prior, ax):
    # ensemble average
    vx_avg = prior.compute_forward_error(ax=ax)
    A_avg = prior.compute_free_energy(ax=ax)
    tx0 = prior.second_moment()
    mx_avg = tx0 - vx_avg
    return dict(vx_avg=vx_avg, mx_avg=mx_avg, A_avg=A_avg)


def check_prior_BO_BN_high_dim(prior, n_samples):
    ax_values = np.linspace(1, 3, 30)
    # for `n_samples` instances
    df = simple_run_experiments(
        get_prior_BO_BN_instance, prior=prior,
        ax=ax_values, sample=np.arange(n_samples)
    ).drop(columns=["prior", "sample"])
    # mean over the `n_samples` instances
    df = df.groupby("ax").mean().reset_index()
    # ensemble average
    df_avg = simple_run_experiments(
        get_prior_BO_BN_ensemble, prior=prior, ax=ax_values
    )
    # merging
    df = df.merge(df_avg, on="ax")
    return df


def plot_prior_BO_BN_high_dim(prior, n_samples):
    df = check_prior_BO_BN_high_dim(prior, n_samples)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    axs[0].plot(df["ax"], df["A_avg"], "-", label=r"$\overline{A}$")
    axs[0].plot(df["ax"], df["A"], "x", label=r"$A$")
    axs[0].set(xlabel=r"$a_x^-$")
    axs[0].legend()
    axs[1].plot(df["ax"], df["mx_avg"], "-", label=r"$\overline{m}_x$")
    axs[1].plot(df["ax"], df["mx"], "x", label=r"$m_x$")
    axs[1].plot(df["ax"], df["qx"], "+", label=r"$q_x$")
    axs[1].set(xlabel=r"$a_x^-$")
    axs[1].legend()
    axs[2].plot(df["ax"], df["vx_avg"], "-", label=r"$\overline{v}_x$")
    axs[2].plot(df["ax"], df["vx"], "x", label=r"$v_x$")
    axs[2].plot(df["ax"], df["mse_x"], "+", label=r"$mse_x$")
    axs[2].set(xlabel=r"$a_x^-$")
    axs[2].legend()
    fig.suptitle(prior)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_prior_RS_BN_instance(teacher, student, mx_hat, qx_hat, tx_hat, sample):
    assert student.size == teacher.size, "Teacher and student have different sizes"
    # sample
    x_true = teacher.sample()
    noise = np.random.standard_normal(student.size)
    bx = mx_hat*x_true + np.sqrt(qx_hat)*noise
    ax = qx_hat + tx_hat
    # single instance
    rx, vx = student.compute_forward_posterior(ax, bx)
    vx = np.mean(vx)
    mx = np.mean(x_true*rx)
    qx = np.mean(rx**2)
    mse_x = np.mean((x_true-rx)**2)
    A = student.compute_log_partition(ax, bx)
    return dict(vx=vx, mx=mx, qx=qx, mse_x=mse_x, A=A)


def get_prior_RS_BN_ensemble(teacher, student, mx_hat, qx_hat, tx_hat):
    # ensemble average
    ax = qx_hat + tx_hat
    vx_avg, mx_avg, qx_avg = student.compute_forward_vmq_RS(
        ax=ax, mx_hat=mx_hat, qx_hat=qx_hat, teacher=teacher, tx0_hat=0
    )
    A_avg = student.compute_potential_RS(
        ax=ax, mx_hat=mx_hat, qx_hat=qx_hat, teacher=teacher, tx0_hat=0
    )
    tx0 = teacher.second_moment()
    mse_x_avg = tx0 - 2*mx_avg + qx_avg
    return dict(
        vx_avg=vx_avg, mx_avg=mx_avg, qx_avg=qx_avg,
        mse_x_avg=mse_x_avg, A_avg=A_avg
    )


def check_prior_RS_BN_high_dim(teacher, student, n_samples):
    mx_hat_values = np.linspace(1, 3, 30)
    # for `n_samples` instances
    df = simple_run_experiments(
        get_prior_RS_BN_instance, teacher=teacher, student=student,
        mx_hat=mx_hat_values, qx_hat=1, tx_hat=1, sample=np.arange(n_samples)
    ).drop(columns=["student", "teacher", "sample"])
    # mean over the `n_samples` instances
    df = df.groupby("mx_hat").mean().reset_index()
    # ensemble average
    df_avg = simple_run_experiments(
        get_prior_RS_BN_ensemble, teacher=teacher, student=student,
        mx_hat=mx_hat_values, qx_hat=1, tx_hat=1
    )
    # merging
    df = df.merge(df_avg, on="mx_hat")
    return df


def plot_prior_RS_BN_high_dim(teacher, student, n_samples):
    df = check_prior_RS_BN_high_dim(teacher, student, n_samples)
    fig, axs = plt.subplots(1, 5, figsize=(16, 4), sharex=True)
    axs[0].plot(df["mx_hat"], df["A_avg"], "-", label=r"$\overline{A}$")
    axs[0].plot(df["mx_hat"], df["A"], "x", label=r"$A$")
    axs[0].set(xlabel=r"$\widehat{m}_x^-$")
    axs[0].legend()
    axs[1].plot(df["mx_hat"], df["mx_avg"], "-", label=r"$\overline{m}_x$")
    axs[1].plot(df["mx_hat"], df["mx"], "x", label=r"$m_x$")
    axs[1].set(xlabel=r"$\widehat{m}_x^-$")
    axs[1].legend()
    axs[2].plot(df["mx_hat"], df["qx_avg"], "-", label=r"$\overline{q}_x$")
    axs[2].plot(df["mx_hat"], df["qx"], "x", label=r"$q_x$")
    axs[2].set(xlabel=r"$\widehat{m}_x^-$")
    axs[2].legend()
    axs[3].plot(df["mx_hat"], df["vx_avg"], "-", label=r"$\overline{v}_x$")
    axs[3].plot(df["mx_hat"], df["vx"], "x", label=r"$v_x$")
    axs[3].set(xlabel=r"$\widehat{m}_x^-$")
    axs[3].legend()
    axs[4].plot(df["mx_hat"], df["mse_x_avg"], "-", label=r"$\overline{mse}_x$")
    axs[4].plot(df["mx_hat"], df["mse_x"], "x", label=r"$mse_x$")
    axs[4].set(xlabel=r"$\widehat{m}_x^-$")
    axs[4].legend()
    fig.suptitle(f"teacher={teacher}\nstudent={student}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])


"""CHECK HIGH DIM LIKELIHOOD"""


def get_likelihood_BO_BN_instance(likelihood, az, tau_z, sample):
    assert az > 1 / tau_z, "az must be greater than 1/tau_z"
    # sample
    z_true = np.sqrt(tau_z)*np.random.standard_normal(likelihood.size)
    y = likelihood.sample(z_true)
    noise = np.random.standard_normal(likelihood.size)
    mz_hat = az - 1 / tau_z
    bz = mz_hat*z_true + np.sqrt(mz_hat)*noise
    # single instance
    rz, vz = likelihood.compute_backward_posterior(az, bz, y)
    vz = np.mean(vz)
    mz = np.mean(z_true*rz)
    qz = np.mean(rz**2)
    mse_z = np.mean((z_true-rz)**2)
    A = likelihood.compute_log_partition(az, bz, y)
    return dict(vz=vz, mz=mz, qz=qz, mse_z=mse_z, A=A)


def get_likelihood_BO_BN_ensemble(likelihood, az, tau_z):
    assert az > 1 / tau_z, "az must be greater than 1/tau_z"
    # ensemble average
    vz_avg = likelihood.compute_backward_error(az=az, tau_z=tau_z)
    A_avg = likelihood.compute_free_energy(az=az, tau_z=tau_z)
    mz_avg = tau_z - vz_avg
    return dict(vz_avg=vz_avg, mz_avg=mz_avg, A_avg=A_avg)


def check_likelihood_BO_BN_high_dim(likelihood, n_samples):
    az_values = np.linspace(1, 3, 30)
    # for `n_samples` instances
    df = simple_run_experiments(
        get_likelihood_BO_BN_instance, likelihood=likelihood,
        az=az_values, tau_z=2, sample=np.arange(n_samples)
    ).drop(columns=["likelihood", "sample"])
    # mean over the `n_samples` instances
    df = df.groupby("az").mean().reset_index()
    # ensemble average
    df_avg = simple_run_experiments(
        get_likelihood_BO_BN_ensemble, likelihood=likelihood,
        az=az_values, tau_z=2
    )
    # merging
    df = df.merge(df_avg, on="az")
    return df


def plot_likelihood_BO_BN_high_dim(likelihood, n_samples):
    df = check_likelihood_BO_BN_high_dim(likelihood, n_samples)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    axs[0].plot(df["az"], df["A_avg"], "-", label=r"$\overline{A}$")
    axs[0].plot(df["az"], df["A"], "x", label=r"$A$")
    axs[0].set(xlabel=r"$a_z^+$")
    axs[0].legend()
    axs[1].plot(df["az"], df["mz_avg"], "-", label=r"$\overline{m}_z$")
    axs[1].plot(df["az"], df["mz"], "x", label=r"$m_z$")
    axs[1].plot(df["az"], df["qz"], "+", label=r"$q_z$")
    axs[1].set(xlabel=r"$a_z^+$")
    axs[1].legend()
    axs[2].plot(df["az"], df["vz_avg"], "-", label=r"$\overline{v}_z$")
    axs[2].plot(df["az"], df["vz"], "x", label=r"$v_z$")
    axs[2].plot(df["az"], df["mse_z"], "+", label=r"$mse_z$")
    axs[2].set(xlabel=r"$a_z^+$")
    axs[2].legend()
    fig.suptitle(likelihood)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_likelihood_RS_BN_instance(teacher, student,
                                  mz_hat, qz_hat, tz_hat, tz0, sample):
    assert student.size == teacher.size, "Teacher and student have different sizes"
    # sample
    z_true = np.sqrt(tz0)*np.random.standard_normal(teacher.size)
    y = teacher.sample(z_true)
    noise = np.random.standard_normal(student.size)
    bz = mz_hat*z_true + np.sqrt(qz_hat)*noise
    az = qz_hat + tz_hat
    # single instance
    rz, vz = student.compute_backward_posterior(az, bz, y)
    vz = np.mean(vz)
    mz = np.mean(z_true*rz)
    qz = np.mean(rz**2)
    mse_z = np.mean((z_true-rz)**2)
    A = student.compute_log_partition(az, bz, y)
    return dict(vz=vz, mz=mz, qz=qz, mse_z=mse_z, A=A)


def get_likelihood_RS_BN_ensemble(teacher, student,
                                  mz_hat, qz_hat, tz_hat, tz0):
    tz0_hat = 1 / tz0
    # ensemble average
    az = qz_hat + tz_hat
    vz_avg, mz_avg, qz_avg = student.compute_backward_vmq_RS(
        az=az, mz_hat=mz_hat, qz_hat=qz_hat, teacher=teacher, tz0_hat=tz0_hat
    )
    A_avg = student.compute_potential_RS(
        az=az, mz_hat=mz_hat, qz_hat=qz_hat, teacher=teacher, tz0_hat=tz0_hat
    )
    mse_z_avg = tz0 - 2*mz_avg + qz_avg
    return dict(
        vz_avg=vz_avg, mz_avg=mz_avg, qz_avg=qz_avg,
        mse_z_avg=mse_z_avg, A_avg=A_avg
    )


def check_likelihood_RS_BN_high_dim(teacher, student, n_samples):
    mz_hat_values = np.linspace(1, 3, 30)
    samples = np.arange(n_samples)
    # for `n_samples` instances
    df = simple_run_experiments(
        get_likelihood_RS_BN_instance, teacher=teacher, student=student,
        mz_hat=mz_hat_values, qz_hat=1, tz_hat=1, tz0=1, sample=samples
    ).drop(columns=["student", "teacher", "sample"])
    # mean over the `n_samples` instances
    df = df.groupby("mz_hat").mean().reset_index()
    # ensemble average
    df_avg = simple_run_experiments(
        get_likelihood_RS_BN_ensemble, teacher=teacher, student=student,
        mz_hat=mz_hat_values, qz_hat=1, tz_hat=1, tz0=1
    )
    # merging
    df = df.merge(df_avg, on="mz_hat")
    return df


def plot_likelihood_RS_BN_high_dim(teacher, student, n_samples):
    df = check_likelihood_RS_BN_high_dim(teacher, student, n_samples)
    fig, axs = plt.subplots(1, 5, figsize=(16, 4), sharex=True)
    axs[0].plot(df["mz_hat"], df["A_avg"], "-", label=r"$\overline{A}$")
    axs[0].plot(df["mz_hat"], df["A"], "x", label=r"$A$")
    axs[0].set(xlabel=r"$\widehat{m}_z^+$")
    axs[0].legend()
    axs[1].plot(df["mz_hat"], df["mz_avg"], "-", label=r"$\overline{m}_z$")
    axs[1].plot(df["mz_hat"], df["mz"], "x", label=r"$m_z$")
    axs[1].set(xlabel=r"$\widehat{m}_z^+$")
    axs[1].legend()
    axs[2].plot(df["mz_hat"], df["qz_avg"], "-", label=r"$\overline{q}_z$")
    axs[2].plot(df["mz_hat"], df["qz"], "x", label=r"$q_z$")
    axs[2].set(xlabel=r"$\widehat{m}_z^+$")
    axs[2].legend()
    axs[3].plot(df["mz_hat"], df["vz_avg"], "-", label=r"$\overline{v}_z$")
    axs[3].plot(df["mz_hat"], df["vz"], "x", label=r"$v_z$")
    axs[3].set(xlabel=r"$\widehat{m}_z^+$")
    axs[3].legend()
    axs[4].plot(df["mz_hat"], df["mse_z_avg"], "-", label=r"$\overline{mse}_z$")
    axs[4].plot(df["mz_hat"], df["mse_z"], "x", label=r"$mse_z$")
    axs[4].set(xlabel=r"$\widehat{m}_z^+$")
    axs[4].legend()
    fig.suptitle(f"teacher={teacher}\nstudent={student}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])

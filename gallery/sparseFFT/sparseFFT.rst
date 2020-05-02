.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_.._gallery_sparseFFT_sparseFFT.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_.._gallery_sparseFFT_sparseFFT.py:


Sparse gradient
===============



.. code-block:: default

    from tramp.algos import EarlyStoppingEP
    from tramp.variables import SISOVariable as V, SILeafVariable as O, MILeafVariable, SIMOVariable
    from tramp.channels import DFTChannel, GaussianChannel
    from tramp.priors import GaussBernouilliPrior, GaussianPrior
    from tramp.experiments import TeacherStudentScenario
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rc, cm
    rc('text', usetex=True)
    rc('font', family='serif')
    #



Define a Sparse FFT teacher


.. code-block:: default



    class SparseFFT_Teacher():
        def __init__(self, size, noise_var):
            self.t = np.linspace(-2*np.pi, 2*np.pi, size, endpoint=False)
            self.channel = GaussianChannel(var=noise_var)

        def sample(self, seed=None, verbose=False):
            if seed:
                np.random.seed(seed)
            x = np.cos(self.t) + np.sin(2*self.t)
            z = DFTChannel(real=True).sample(x)
            nonzero = (np.abs(z) > 1e-11)
            y = self.channel.sample(x)
            if verbose:
                print(
                    f"SparseFFT_Teacher: var(x) = {x.var(): .3f} "
                    f"rho(z)={nonzero.mean(): .3f} var(z)={z[nonzero].var(): .3f}"
                )
            return {"x": x, "z": z, "y": y}



Define a sparse FFT student


.. code-block:: default


    def build_sparse_fft_student(size, prior_var, fft_rho, fft_var, noise_var):
        x_shape = (size,)
        fft_shape = (2,) + x_shape
        student = (
            GaussianPrior(size=size, var=prior_var) @
            SIMOVariable(id="x", n_next=2) @ (
                GaussianChannel(var=noise_var) @ O("y") + (
                    DFTChannel(real=True) +
                    GaussBernouilliPrior(size=fft_shape, var=fft_var, rho=fft_rho)
                ) @
                MILeafVariable(id="z", n_prev=2)
            )
        ).to_model()
        return student


Plotting function


.. code-block:: default



    def plot_sparse_fft(dic, save_fig=False, block=False):
        _, axes = plt.subplots(1, 3, figsize=(16, 6))
        cmap = cm.get_cmap('plasma_r')
        tab_col = plt.rcParams['axes.prop_cycle'].by_key()['color']
        tab_l1, tab_l2,  tab_l3 = [], [], []

        l, = axes[0].plot(dic['y'], 'o', color=tab_col[2], label=r'$y$')
        tab_l1.append(l)

        l, = axes[1].plot(dic['x']["x"],
                          color=tab_col[0], label=r'$x^*$')
        tab_l2.append(l)
        l, = axes[1].plot(dic['x_pred']["x"],
                          color=tab_col[1], label=r'$\hat{x}$')
        tab_l2.append(l)

        l, = axes[2].plot(dic['x']["z"][0].ravel(),
                          color=tab_col[0], label=r"$\textrm{Re}(z^*)$")
        tab_l3.append(l)
        l, = axes[2].plot(dic['x_pred']["z"][0].ravel(),
                          color=tab_col[1], label=r"$\textrm{Re}(\hat{z})$")
        tab_l3.append(l)
        l, = axes[2].plot(dic['x']["z"][1].ravel(), '--',
                          color=tab_col[0], label=r"$\textrm{Im}(z^*)$")
        tab_l3.append(l)
        l, = axes[2].plot(dic['x_pred']["z"][1].ravel(), '--',
                          color=tab_col[1], label=r"$\textrm{Im}(\hat{z})$")
        tab_l3.append(l)

        """ Titles  """
        axes[0].set_title(r'$y$')
        axes[1].set_title(r'$x$')
        axes[2].set_title(r'$z = \textrm{DFT}(x)$')

        """ Ticks   """
        axes[0].set_xlim([0, dic['N']])
        axes[1].set_xlim([0, dic['N']])
        axes[2].set_xlim([0, dic['N']/4])

        axes[1].legend(tab_l2, [l.get_label() for l in tab_l2], loc='lower center', fancybox=True,
                       shadow=False, ncol=1)
        axes[0].legend(tab_l1, [l.get_label() for l in tab_l1], loc='lower center', fancybox=True,
                       shadow=False, ncol=1)
        axes[2].legend(tab_l3, [l.get_label() for l in tab_l3], loc='lower center', fancybox=True,
                       shadow=False, ncol=1)

        """ Save   """
        if save_fig:
            dir_fig = 'Figures/'
            os.makedirs(dir_fig) if not os.path.exists(dir_fig) else 0
            file_name = f'{dir_fig}{dic["model"]}_N={dic["N"]}_rho={str(dic["rho"]).replace(".","")}_seed={dic["seed"]}.pdf'

            plt.tight_layout()
            plt.savefig(file_name, format='pdf', dpi=1000,
                        bbox_inches="tight", pad_inches=0.1)

        """ Show   """
        plt.show()


Parameters


.. code-block:: default



    # Size #
    N = 100
    # Sparsity #
    rho = 0.02
    # Seed #
    seed = 1



Build the teacher


.. code-block:: default

    teacher = SparseFFT_Teacher(size=N, noise_var=0.1)

    # Build the student
    student = build_sparse_fft_student(
        size=N,  prior_var=1, fft_rho=rho, fft_var=18, noise_var=0.1)

    # Create a Teacher Student Scenario
    # Variables to track #
    x_ids = ["x", "z"]
    scenario = TeacherStudentScenario(teacher, student, x_ids=x_ids)
    scenario.setup(seed=seed)


Run EP


.. code-block:: default


    # Max iter #
    max_iter = 1000
    # Damping value #
    damping = 0.1

    scenario.run_ep(max_iter=max_iter,
                    damping=damping,
                    callback=EarlyStoppingEP(tol=1e-2)
                    )
    dic = {'model': 'sparse_fft', 'N': N, 'rho': rho, 'seed': seed,
           'y': scenario.observations["y"], 'x': scenario.x_true, 'x_pred': scenario.x_pred}


Plot


.. code-block:: default

    plot_sparse_fft(dic, save_fig=False)




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_.._gallery_sparseFFT_sparseFFT.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: sparseFFT.py <sparseFFT.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: sparseFFT.ipynb <sparseFFT.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

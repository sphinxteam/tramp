Computing new modules
=====================

.. tip::
   To implement a new module the main difficulty is to compute the log-partition
..

Variable
--------

An approximate belief, which we may as well call a variable type, is specified by 
the base space :math:`X` as well as the chosen set of sufficient statistics :math:`\phi(x)`. Any variable type defines an exponential family distribution 

.. math::

   \begin{equation}
      p(x|\lambda) = e^{\lambda^\intercal  \phi(x) - A[\lambda]}
   \end{equation}


indexed by the natural parameter :math:`\lambda`. The log-partition, the mean and the variance are then given by:

.. math::

   \begin{align}
      A[\lambda] &= \ln \int_X dx e^{\lambda^\intercal  \phi(x)} \\
      r[\lambda] &= \partial_b A[\lambda], \\
      v[\lambda] &= \partial_b^2 A[\lambda],
   \end{align}


Prior
-----

Let :math:`f(x) = p_0(x) = \prod_{i=1}^N p_0(x^{(i)})` be a separable prior over :math:`x \in \mathbb{R}^N`. The log-partition, posterior means and variances  are given by:

.. math::

   \begin{align}
      &A_f[a_{x \rightarrow f} b_{x \rightarrow f}] = \sum_{i=1}^{N} A_f[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}], \\
      \label{separable_rf}
      &r_x^{f(i)}[a_{x \rightarrow f} b_{x \rightarrow f}] = r_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}], \\
      \label{separable_vf_prior}
      &v_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}] = \frac{1}{N} \sum_{i=1}^{N}
      v_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}].
   \end{align}


Channel
-------

The factor :math:`f(x, z) = p(x|z) = \delta(x - Wz)` is the deterministic channel :math:`x = Wz`. Unless explicitly specified, we will always consider isotropic Gaussian beliefs on both :math:`x` and :math:`z`. The log-partition is given by

.. math::

   \begin{align}
      \mathcal{A}[a_x, b_x, a_z, b_z] &\equiv \log \int \int d_x  d_z p(x|z) e^{-\frac{1}{2}a_x x^2 + b_x x} e^{-\frac{1}{2}a_z z^2 + b_z z}\\
      r_x &= \partial_{b_x}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      v_x &= \partial^2_{b_x}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      r_z &= \partial_{b_z}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      v_z &= \partial^2_{b_z}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
   \end{align}
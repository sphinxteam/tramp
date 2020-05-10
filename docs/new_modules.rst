Creating new modules
====================


Variable
--------

An approximate belief, which we may as well call a variable type, is specified by
the base space $X$ as well as the chosen set of sufficient statistics $\phi(x)$.
Any variable type defines an exponential family distribution

.. math::

      p(x|\lambda) = e^{\lambda^\intercal  \phi(x) - A[\lambda]}


indexed by the natural parameter $\lambda$. The log-partition, the mean and the variance are then given by:

.. math::

      A[\lambda] &= \ln \int_X dx e^{\lambda^\intercal  \phi(x)} \\
      r[\lambda] &= \partial_b A[\lambda], \\
      v[\lambda] &= \partial_b^2 A[\lambda],



Prior
-----

General definitions
___________________

Let $f(x) = p_0(x) = \prod_{i=1}^N p_0(x^{(i)})$ be a separable prior over $x \in \mathbb{R}^N$.
The log-partition, posterior means and variances  are given by:

.. math::

      &A_f[a_{x \rightarrow f} b_{x \rightarrow f}] = \sum_{i=1}^{N} A_f[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}], \\
      \label{separable_rf}
      &r_x^{f(i)}[a_{x \rightarrow f} b_{x \rightarrow f}] = r_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}], \\
      \label{separable_vf_prior}
      &v_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}] = \frac{1}{N} \sum_{i=1}^{N}
      v_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}].

with 



Natural prior
_____________

In the case of a natural prior with parameters $\lambda_0$

.. math::
  &A_f[\lambda_{z \rightarrow f}] =
  A_z[\lambda_{z \rightarrow f} + \lambda_0] - A_z[\lambda_0], \\
  &\mu_z^f[\lambda_{z \rightarrow f}] = \mu_z[\lambda_{z  \rightarrow f} + \lambda_0],




Example: exponential prior
^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us illustrate with the natural prior $p_0(x | b_0) = \exp(b_0 x - A_z[b_0]) 1[x>0]$, where $b_0<0$. The natural parameters $\lambda_0 = {a_0, b_0} = {0, b_0}$ 

.. math::

      A_f[a_{x \rightarrow f} b_{x \rightarrow f}] &= \sum_{i=1}^{N} A_f[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}]\\
      A_f[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}]  &\equiv \log \int_{\mathbb{R}} p_0(x | b_0) e^{-\frac{1}{2} a_{x \rightarrow f} x^2 + b_{x \rightarrow f}^{(i)} x} = \log \int_{\mathbb{R}_+} e^{-\frac{1}{2} a_{x \rightarrow f} x^2 + (b_{x \rightarrow f}^{(i)} + b_0) x} - A_z[b_0] = A_+[a b^{(i)}] - A_z[b_0]

with $a = a_{x \rightarrow f}$ and $b^{(i)} = b_{x \rightarrow f}^{(i)} + b_0$ and

.. math::
      
      \begin{cases}
      A_+[a b^{(i)}] &\equiv  \log \int_{\mathbb{R}_+} e^{-\frac{1}{2} a_{x \rightarrow f} x^2 + b^{(i)} x} = A[a b^{(i)}] + \log p_+[a b^{(i)} ] \\
      A_z[b_0] &\equiv  \log \left(\int_0^{\infty } \exp (b_0 x) \, dx\right) = \log \left(-\frac{1}{b_0}\right)
      \end{cases}

with 

.. math::
      
      \begin{cases}
      A[a b^{(i)}] &=\ln \int dx\, e^{-\frac{1}{2} ax^2 + b^{(i)} x} = \frac{(b^{(i)})^2}{2a} + \frac{1}{2} \ln\frac{2\pi}{a} \\
      p_+[a b^{(i)}] &= \int_{\mathbb{R}_+} dx \mathcal{N}(x|ab) \equiv \Phi(z_+) \quad \text{with} \quad z_+ =  \frac{b^{(i)}}{\sqrt{a}}, 
      \end{cases}



Then the moments are given by:

.. math::

      r_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}] &= r_+[a b^{(i)}] =  + \frac{1}{\sqrt{a}} \left\{z_+ +  \frac{\mathcal{N}(z_+)}{\Phi(z_+)}\right\}, \\ 
      v_x^{f}[a_{x \rightarrow f} b_{x \rightarrow f}^{(i)}] &= v_+[a b^{(i)}] = \frac{1}{a} \left\{ 1 - \frac{z_+ \mathcal{N}(z_+)}{\Phi(z_+)} -  \frac{\mathcal{N}(z_+)^2}{\Phi(z_+)^2} \right\}.



Channel
-------

The factor $f(x, z) = p(x|z) = \delta(x - Wz)$ is the deterministic channel $x = Wz$.
Unless explicitly specified, we will always consider isotropic Gaussian beliefs on both $x$ and $z$.
The log-partition is given by

.. math::

      \mathcal{A}[a_x, b_x, a_z, b_z] &\equiv \log \int \int db_x  db_z p(x|z) e^{-\frac{1}{2}a_x x^2 + b_x x} e^{-\frac{1}{2}a_z z^2 + b_z z}\\
      r_x &= \partial_{b_x}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      v_x &= \partial^2_{b_x}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      r_z &= \partial_{b_z}  \mathcal{A}[a_x, b_x, a_z, b_z] \\
      v_z &= \partial^2_{b_z}  \mathcal{A}[a_x, b_x, a_z, b_z]



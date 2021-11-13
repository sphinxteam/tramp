.. currentmodule:: tramp.priors

.. _api_priors:

Priors
======

.. _gaussian_prior:

Gaussian
--------

.. autoclass:: GaussianPrior

----------

**Definition** The Gaussian prior belongs to the :ref:`normal` exponential family distibution

.. math:: p(x) =  \mathcal{N}(x | r, v) = p(x | a_0, b_0)

with associated natural parameter :math:`a_0 = \frac{1}{v}, b_0 = \frac{r}{v}`

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A(a,b) - A(a_0, b_0), \quad
  r_x[a_x^-,b_x^-] = r(a,b), \quad
  v_x[a_x^-,b_x^-] = v(a,b)

where $A(a,b), r(a,b), v(a,b)$ are the log-partition, mean and variance of the
:ref:`normal` belief and $a = a_x^- + a_0, b = b_x^- + b_0$.

A special property of the Gaussian prior is that
the output messages, sent from the prior $p$ to the variable $x$, are constant
$a_{p \rightarrow x} = a_0, b_{p \rightarrow x} = b_0$.
The corresponding :meth:`compute_forward_message` method,
implemented in the base :class:`Prior` class, is overwritten
to directly send these constant messages.


----------

**State evolution** The measures over $b_x^-$ are the Gaussian measures:

.. math::
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} \phi(b_x^-) = \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )} \phi(b_x^-) \\
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} x \phi(b_x^-) = \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )} r_x^* \phi(b_x^-)

where:

.. math::
  &\tilde{a}_0 = a_0 + \hat{\tau}_x^{-(0)}, \quad
  \tilde{v} = \frac{1}{\tilde{a}_0}, \quad
  \tilde{r} = \frac{b_0}{\tilde{a}_0}, \\
  &b_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} b_x^-, \quad
  a_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} \hat{m}_x^-, \quad
  r_x^* = \frac{b_0 + b_x^{-*}}{\tilde{a}_0 + a_x^{-*}}

A special property of the Gaussian prior is that
the output messages, sent from the prior $p$ to the variable $x$, are constant
$a_{p \rightarrow x} = a_0$. The corresponding methods:

- :meth:`compute_forward_state_evolution` in a Bayes-optimal setting + Bayesian network context,
- :meth:`compute_forward_state_evolution_BO` in a Bayes-optimal setting + factor graph context,

implemented in the base :class:`Prior` class, are overwritten to directly send
these constant messages.

.. todo:: RS mismatched case GaussianPrior


.. _binary_prior:

Binary
------

.. autoclass:: BinaryPrior

----------

**Definition** The binary prior belongs to the :ref:`binary` exponential family distibution

.. math:: p(x) =  p_+ \delta_+(x) + p_- \delta_-(x) = p(x | b_0)

with associated natural parameter :math:`b_0 = \tfrac{1}{2}\ln\tfrac{p_+}{p_-}`

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A(b) - A(b_0) - \tfrac{1}{2} a_x^-, \quad
  r_x[a_x^-,b_x^-] = r(b), \quad
  v_x[a_x^-,b_x^-] = v(b)

where $A(b), r(b), v(b)$ are the log-partition, mean and variance of the
:ref:`binary` belief and $b = b_x^- + b_0$.

----------

**State evolution** The measures over $b_x^-$ are given by linear combinations of Gaussian measures:

.. math::
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} \phi(b_x^-) = \sum_{s = \pm} p_s
  \mathbb{E}_{\mathcal{N}(b_x^- | \hat{m}_x^- s, \hat{q}_x^-)} \phi(b_x^-) \\
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} x \phi(b_x^-) = \sum_{s = \pm} p_s
  \mathbb{E}_{\mathcal{N}(b_x^- | \hat{m}_x^- s, \hat{q}_x^-)} s \phi(b_x^-)


.. _sparse_prior:

Gauss-Bernoulli
---------------

.. autoclass:: GaussBernoulliPrior

----------

**Definition** The Gauss-Bernouilli prior belongs to the :ref:`sparse`
exponential family distibution

.. math:: p(x)=[1-\rho]\delta(x) + \rho\mathcal{N}(x|r,v)=p(x|a_0, b_0, \eta_0)

with associated natural parameters

.. math::
  a_0 = \tfrac{1}{v}, \quad
  b_0 = \tfrac{r}{v}, \quad
  \eta_0 = A(a_0, b_0) - \ln \tfrac{\rho}{1-\rho}

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A(a,b,\eta_0) - A(a_0, b_0, \eta_0), \quad
  r_x[a_x^-,b_x^-] = r(a,b,\eta_0), \quad
  v_x[a_x^-,b_x^-] = v(a,b,\eta_0)

where $A(a,b,\eta), r(a,b,\eta), v(a,b,\eta)$ are the log-partition, mean and variance of the
:ref:`sparse` belief and $a = a_x^- + a_0, b = b_x^- + b_0$.

----------

**State evolution** The measures over $b_x^-$ are given by linear combinations of Gaussian measures:

.. math::
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} \phi(b_x^-) =
  [1-\tilde{\rho}] \, \mathbb{E}_{\mathcal{N}(b_x^- | 0, \hat{q}_x^-)}\phi(b_x^-)
  + \tilde{\rho} \, \mathbb{E}_{\mathcal{N}(
  b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )} \phi(b_x^-) \\
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} x \phi(b_x^-) =
  \tilde{\rho} \, \mathbb{E}_{\mathcal{N}(
  b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )}  r_x^* \phi(b_x^-)

where:

.. math::
  &\tilde{a}_0 = a_0 + \hat{\tau}_x^{-(0)}, \quad
  \tilde{v} = \frac{1}{\tilde{a}_0}, \quad
  \tilde{r} = \frac{b_0}{\tilde{a}_0}, \quad
  \tilde{\rho} = p(\tilde{a}_0 b_0 \eta_0), \\
  &b_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} b_x^-, \quad
  a_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} \hat{m}_x^-, \quad
  r_x^* = \frac{b_0 + b_x^{-*}}{\tilde{a}_0 + a_x^{-*}}


.. _mixture_prior:

Gaussian Mixture
-----------------

.. autoclass:: GaussianMixturePrior

----------

**Definition** The Gaussian Mixture prior belongs to the :ref:`mixture`
exponential family distibution

.. math:: p(x)=\sum_k p_k \mathcal{N}(x|r_k,v_k)= p(x|a_0, b_0, \eta_0)

with associated natural parameters $a_0 = \{a_k\}_{k=1}^K$,
$b_0 = \{b_k\}_{k=1}^K$, $\eta_0 = \{\eta_k\}_{k=1}^K$ given by:

.. math::
  a_k = \tfrac{1}{v_k}, \quad
  b_k = \tfrac{r_k}{v_k}, \quad
  \eta_k = \ln p_k - A(a_k b_k).

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A(a,b,\eta_0) - A(a_0, b_0, \eta_0), \quad
  r_x[a_x^-,b_x^-] = r(a,b,\eta_0), \quad
  v_x[a_x^-,b_x^-] = v(a,b,\eta_0)

where $A(a,b,\eta), r(a,b,\eta), v(a,b,\eta)$ are the log-partition, mean and variance of the
:ref:`mixture` belief and $a = a_x^- + a_0, b = b_x^- + b_0$.

----------

**State evolution** The measures over $b_x^-$ are given by linear combinations of Gaussian measures:

.. math::
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} \phi(b_x^-) = \sum_k \tilde{p}_k
  \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}_k, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}_k
  )} \phi(b_x^-) \\
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} x \phi(b_x^-) = \sum_k \tilde{p}_k
  \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}_k, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}_k
  )}  r_x^k \phi(b_x^-)

where:

.. math::
  &\tilde{a}_k = a_k + \hat{\tau}_x^{-(0)}, \quad
  \tilde{v}_k = \frac{1}{\tilde{a}_k}, \quad
  \tilde{r}_k = \frac{b_k}{\tilde{a}_k}, \quad
  \tilde{p}_k = p_k(\tilde{a}_0 b_0 \eta_0), \\
  &b_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} b_x^-, \quad
  a_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} \hat{m}_x^-, \quad
  r_x^k = \frac{b_k + b_x^{-*}}{\tilde{a}_k + a_x^{-*}}


.. _positive_prior:

Positive
--------

.. autoclass:: PositivePrior

----------

**Definition** The Positive prior belongs to the :ref:`positive`
exponential family distibution

.. math:: p(x)=  2 * 1_+(x) \mathcal{N}(x|0, 1) = p(x|a_0, b_0)

with associated natural parameters $a_0 = 1, b_0 = 0$.

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A_+(a,b) - A_+(a_0, b_0), \quad
  r_x[a_x^-,b_x^-] = r(a,b), \quad
  v_x[a_x^-,b_x^-] = v(a,b)

where $A_+(a,b), r_+(a,b), v_+(a,b)$ are the log-partition, mean and variance of the
:ref:`positive` belief and $a = a_x^- + a_0, b = b_x^- + b_0$.

----------

**State evolution** The measures over $b_x^-$ are given by the Gaussian measures:

.. math::
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} \phi(b_x^-) =
  \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )} \frac{p_+(a,b)}{p_+(\tilde{a}_0 b_0)} \phi(b_x^-) \\
  &\mathbb{E}_{p^{(0)}(b_x^-, x)} x \phi(b_x^-) =
  \mathbb{E}_{\mathcal{N}(
    b_x^- | \hat{m}_x^- \tilde{r}, \hat{q}_x^- + (\hat{m}_x^-)^2 \tilde{v}
  )}  \frac{p_+(a,b)}{p_+(\tilde{a}_0 b_0)} r_x^* \phi(b_x^-)

where:

.. math::
  &\tilde{a}_0 = a_0 + \hat{\tau}_x^{-(0)}, \quad
  \tilde{v} = \frac{1}{\tilde{a}_0}, \quad
  \tilde{r} = \frac{b_0}{\tilde{a}_0}, \\
  &b_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} b_x^-, \quad
  a_x^{-*} = \frac{\hat{m}_x^-}{\hat{q}_x^-} \hat{m}_x^-, \quad
  r_x^* = \frac{b_0 + b_x^{-*}}{\tilde{a}_0 + a_x^{-*}}

and :math:`p_+(a,b)` denotes the probability that a normal belief with natural
parameters $ab$ falls within the $\mathbb{R}_+$ interval, which is implemented
by the function :func:`p` of the :ref:`positive` belief.


.. _exponential_prior:

Exponential
-----------

.. autoclass:: ExponentialPrior

----------

**Definition** The Exponential prior belongs to the :ref:`exponential` belief
family distibution

.. math:: p(x)= 1_+(x) \tfrac{1}{r} e^{-\frac{x}{r}} = p(x|b_0)

with associated natural parameter $b_0 = -\frac{1}{r}$.

----------

**Expectation propagation** The scalar log-partition, posterior mean and variance are given by:

.. math::
  A_p[a_x^-,b_x^-] = A_+(a,b) - A(b_0), \quad
  r_x[a_x^-,b_x^-] = r_+(a,b), \quad
  v_x[a_x^-,b_x^-] = v_+(a,b)

where $A_+(a,b), r_+(a,b), v_+(a,b)$ are the log-partition, mean and variance of the
:ref:`positive` belief with $a = a_x^- + a_0, b = b_x^- + b_0$ and
$A(b_0) = -\ln(-b_0)$ is the log-partition of the :ref:`exponential` belief.

----------

**State evolution**

.. todo:: b and bx measures ExponentialPrior


.. _map_L1_prior:

MAP L1 norm
-----------

.. autoclass:: MAP_L1NormPrior


----------

**Definition** It corresponds to the prior obtained by the MAP
approximation applied to the factor $f(x) = e^{-E(x)}$ with a L1 penalty

.. math:: E(x) = \gamma \Vert x \Vert_1

where $\gamma$ is the regularization parameter. The penalty is
separable $E(x) = \sum_{i=1}^{N_x} E(x^{[i]})$.

----------

**Expectation propagation**
The scalar log-partition, posterior mean and variance are given by the usual
formula for :ref:`MAP priors <map_priors_scalar>` using the scalar proximal operator.
For the $\Vert . \Vert_1$ penalty, the scalar proximal operator is known as
the soft thresholding operator:

.. math::
  \mathrm{prox}_{\gamma |.|} (x) = \max\left( 0, 1-\frac{\gamma}{|x|} \right) x.


.. _map_L21_prior:

MAP L21 norm
------------

.. autoclass:: MAP_L21NormPrior

----------

**Definition** It corresponds to the prior obtained by the MAP
approximation applied to the factor $f(x) = e^{-E(x)}$ with a L21 penalty

.. math:: E(x) = \gamma \Vert x \Vert_{2,1}

where $\gamma$ is the regularization parameter. You must specify over which
axis the L2 norm is taken. For example, say the variable
$x \in \mathbb{R}^{d \times P}$ is an array of shape $(d, P)$ and we take the
L2 norm over the ``axis=0``, then the L21 norm is precisely

.. math::
  \Vert x \Vert_{2,1} = \sum_{j=1}^P \Vert x_j \Vert_2 =
  \sum_{j=1}^P \sqrt{ \sum_{i=1}^d x_{ij}^2   }

The penalty is **not** separable due to the L2 norm.

----------

**Expectation propagation**
The vectorized log-partition, posterior mean and variance are given by the usual
formula for :ref:`MAP priors <map_priors_vectorized>` using the proximal operator.
For the $\Vert . \Vert_{2,1}$ penalty, the proximal operator is known as
the group soft thresholding operator:

.. math::
  \mathrm{prox}_{\gamma\Vert .\Vert_{2,1}} (x) =
  \max\left( 0, 1 - \frac{\gamma}{\Vert x_j \Vert_2} \right) x.


.. _committee_binary_prior:

Committee Binary
----------------

.. autoclass:: CommitteeBinaryPrior

----------

**Definition** The binary prior belongs to the :ref:`binary` exponential family distibution

.. math:: p(x) =  p_+ \delta_+(x) + p_- \delta_-(x) = p(x | b_0)

with associated natural parameter :math:`b_0 = \tfrac{1}{2}\ln\tfrac{p_+}{p_-}`.


----------

**Expectation propagation**  The K-committee log-partition, posterior mean and
covariance are given by:

.. math::
  A_p[a_x^-,b_x^-] &= \frac{1}{K} \ln \sum_{x \in (\pm)^K} e^{A_x} - A(b_0), \\
  r_x[a_x^-,b_x^-] &= \sum_{x \in (\pm)^K} \sigma_x x , \\
  \Sigma_x[a_x^-,b_x^-] &= \sum_{xx' \in (\pm)^K}
  \sigma_x \sigma_{x'} (x-x')(x-x')^T

where $A(b)$ is the log-partition of the :ref:`binary` belief,
$\sigma = \mathrm{softmax}(\{A_x\})$ and for a
spin configuration $x \in (\pm)^K$ we set
$A_x = - \frac{1}{2} x \cdot a_x^- x + (b_0 + b_x^-) \cdot x$.

.. warning::
  The committee binary prior is here implemented as an explicit summation over
  the $2^K$ spin configurations, thus $K$ cannot be too large.

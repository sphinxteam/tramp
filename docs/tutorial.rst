Tutorial
========

.. currentmodule:: tramp

This guide can help you start working with TRAMP.

Building your first model
-------------------------

Creating a variable
^^^^^^^^^^^^^^^^^^^

Create a :class:`variables.SISOVariable` ``V`` of size ``N`` drawn from a :class:`priors.GaussBernouilliPrior` separable prior with sparsity ``rho``

.. nbplot::

    >>> from tramp.variables import SISOVariable as V
    >>> from tramp.priors import GaussBernouilliPrior
    >>> N, rho = 100, 0.1
    >>> prior = GaussBernouilliPrior(size=N, rho=rho)
    >>> prior @ V(id="x")

The opeartor ``@`` allows to assign a variable ``x`` drawn from a distribution ``prior``.


Draw a matrix from an ensemble
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a matrix ``W``, an instance of the :class:`ensembles.GaussianEnsemble` of size `` M \times N ``

.. nbplot::

    >>> from tramp.ensembles import GaussianEnsemble
    >>> M, N = 200, 100
    >>> ensemble = GaussianEnsemble(M, N)
    >>> W = ensemble.generate()


Constructing a linear channel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`channels.LinearChannel` multiplies the above variable ``x`` by the matrix ``W`` and can be casted in a new variable ``z = W x`` with ``W ~ ensemble`` and ``x ~  prior``

.. nbplot::

    >>> from tramp.channels import LinearChannel
    >>> model = prior @ V(id="x") @ LinearChannel(W=W, name='W') @ V(id="z0")


Adding other channels
^^^^^^^^^^^^^^^^^^^^^

As many module can be added as will. Each module must be followed by a variable ``V``.
For example a :class:`channels.BiasChannel` with bias ``b`` followed by a :class:`channels.ReluChannel`.

.. nbplot::

    >>> from tramp.channels import BiasChannel, ReluChannel
    >>> b = 0.1
    >>> model = model @ BiasChannel(bias=b) @ V(id='z')
    >>> model = model @ ReluChannel() @ V(id='h')


Adding observations
^^^^^^^^^^^^^^^^^^^


An observation ``O`` can be added as an instance of the :class:`variables.SILeafVariable`.
This observation can be for example the result of the above `model` whose output goes through a noisy :class:`channels.GaussianChannel` with variance ``var``.

.. nbplot::

    >>> from tramp.variables import SILeafVariable as O
    >>> from tramp.channels import GaussianChannel
    >>> model = model @ GaussianChannel(var=1e-2) @ O(id='y')


Plot the model
^^^^^^^^^^^^^^

.. nbplot::

    >>> model = model.to_model()
    >>> model.plot()



Running Expectation Propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Sparse gradient prior
---------------------

Build the model
^^^^^^^^^^^^^^^

Creating a ``x`` sampled from a :class:`priors.GaussianPrior` with ``n_prev=1`` predecessors and ``n_next=2`` successors

.. nbplot::

    >>> from tramp.variables import SIMOVariable as V
    >>> from tramp.priors import GaussianPrior
    >>> N = 100
    >>> prior_x = GaussianPrior(size=N, var=1) @ V(id="x", n_next=2)

and connect it to a :class:`channels.GaussianChannel` with ``y`` observations and a sparse :class:`channels.GradientChannel` constraint

.. nbplot::

    >>> from tramp.channels import GradientChannel, GaussianChannel
    >>> from tramp.priors import GaussBernouilliPrior
    >>> from tramp.variables import SISOVariable as V, MILeafVariable
    >>> x_shape = (N,)
    >>> grad_shape = (1,) + x_shape
    >>> channel_y = GaussianChannel(var=0.1) @ O("y")
    >>> channel_grad = (GradientChannel(shape=x_shape) + GaussBernouilliPrior(size=grad_shape, rho=0.04)) @ MILeafVariable(id="x'", n_prev=2)
    >>> model = prior_x @ ( channel_y +  channel_grad )
    >>> model = model.to_model()
    >>> model.plot()


Teacher-Student scenario
^^^^^^^^^^^^^^^^^^^^^^^^

In a :class:`experiments.BayesOptimalScenario`, a ``teacher`` generates samples of the variables ``x_ids`` and a  ``student`` tries to recover them.
A ``seed`` can be used for reproducibility of the results.

.. nbplot::

    >>> from tramp.experiments import BayesOptimalScenario
    >>> x_ids = ["x", "x'"]
    >>> scenario = BayesOptimalScenario(model, x_ids=x_ids)
    >>> scenario.setup(seed=42)


Run Expectation Propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

EP can be run with a maximum number of iterations ``max_iter`` and ``damping`` on all variables to help convergence.
Different ``callback`` can be used such as :class:`algos.EarlyStoppingEP` if precision `tol` is reached.

.. nbplot::

    >>> from tramp.algos import EarlyStoppingEP
    >>> scenario.run_ep(max_iter=1000, damping= 0.1, callback=EarlyStoppingEP(tol=1e-2))


To use multiple callbacks at a time, you need to use :class:`algos.JoinCallback`:

.. nbplot::
    >>> from tramp.algos import JoinCallback, EarlyStoppingEP, TrackOverlaps, TrackEstimate
    >>> track_overlap = TrackOverlaps(true_values=scenario.x_true)
    >>> track_estimate = TrackEstimate()
    >>> callback = JoinCallback([track_overlap, track_estimate, EarlyStoppingEP(tol=1e-12)])

To be continued...

.. code-links::

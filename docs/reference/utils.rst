Utils
=====

Model builders
--------------

.. autofunction:: tramp.priors.get_prior
.. autofunction:: tramp.channels.get_channel
.. autofunction:: tramp.likelihoods.get_likelihood
.. autofunction:: tramp.models.glm_generative
.. autofunction:: tramp.models.glm_state_evolution
.. autoclass:: tramp.models.MultiLayerModel

Metrics
-------

.. currentmodule:: tramp.algos.metrics

.. autofunction:: mean_squared_error
.. autofunction:: overlap


Experiments
-----------

.. currentmodule:: tramp.experiments

.. autoclass:: TeacherStudentScenario
.. autoclass:: BayesOptimalScenario
.. autofunction:: run_experiments
.. autofunction:: save_experiments
.. autofunction:: find_critical_alpha
.. autofunction:: qplot


Check gradients
---------------

.. currentmodule:: tramp.checks

.. autofunction:: plot_belief_grad_b
.. autofunction:: plot_prior_grad_BO

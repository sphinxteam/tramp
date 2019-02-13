# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] (tests) modulus posterior : why numerical integration less precise ?
- [ ] check complex and n-array variables
- [ ] check second moment computation for non zero mean signal


## TODO

- test if init with a=espilon is better than zero to avoid warnings
- model.sample() for variable with n_prev > 1 factors (eg TV/sparse gradient)
- BrideVariable
  - [x] message and state evo
- notebooks
  - [ ] channels
  - [x] priors
  - [x] likelihoods
  - [x] ridge regression
  - [x] sparse regression
  - [x] perceptron
  - [ ] probit classification
- tests
  - [x] channel posterior
  - [x] proba belief normalized
  - [x] second_moment
  - [x] priors
  - [x] likelihoods
- EP algo
  - [x] message passing
  - [x] callbacks
  - [x] initial conditions
  - [x] a general MessagePassing class
- explainer
  - [ ] animation of message passing (store daft every step)
  - [ ] evolution of beliefs / state evo
- models
  - [x] general DAG
  - [x] generative / inference mode
  - [x] mutlilayer
  - [x] glm
  - [x] model algebra (compose/concat/duplicate)
  - [x] committee
  - [X] regression with TV
- priors
  - [x] TV prior (using MAP)
- channels
  - [x] modulus
  - [x] ReLU
  - [ ] pow(x, n)
- likelihood
  - [x] modulus
- nodes
  - [x] leaf unobserved variable
- state evolution
  - [x] channels
  - [x] priors
  - [x] likelihood
  - [x] EP algo
- ensemble
  - [x] Haar measure
  - [x] gaussian iid
  - [ ] features generator
- GPs
  - [ ] GP prior
  - [ ] sampling operator channel (function -> vector)
  - [ ] linear operator channel (eg derivative, fourier)
- mutual info / evidence
  - [ ] Reeves formula
  - [ ] EP approximation of evidence

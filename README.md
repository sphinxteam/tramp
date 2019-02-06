# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] (tests) modulus posterior : why numerical integration less precise ?
- [ ] y in daft factor_dag missing

## TODO

- first moment
  - needed for ReluChannel and SumChannel
- BrideVariable
  - [ ] message and state evo
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
  - [ ] committee
  - [ ] regression with TV
- priors
  - [ ] TV prior (using MAP)
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
  - [ ] Haar measure
  - [x] gaussian iid
  - [ ] features generator
- GPs
  - [ ] GP prior
  - [ ] sampling operator channel (function -> vector)
  - [ ] linear operator channel (eg derivative, fourier)
- mutual info / evidence
  - [ ] Reeves formula
  - [ ] EP approximation of evidence

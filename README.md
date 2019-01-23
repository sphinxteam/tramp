# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] (tests) modulus measure not normalized
- [ ] (tests) modulus posterior : why numerical integration less precise ?
- [ ] numerical instability: zerodivision error, nan, overflow
- [ ] implement AbsChannel.beliefs_measure

## TODO

- notebooks
  - [ ] channels
  - [ ] priors
  - [ ] likelihoods
  - [ ] ridge regression
  - [ ] sparse regression
  - [ ] perceptron
  - [ ] probit classification
- initial conditions in EP and SE:
  - [ ] a > 1/tau
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
- priors
  - [ ] TV prior (using MAP)
- channels
  - [ ] modulus
  - [ ] ReLU
  - [ ] pow(x, 2)
- likelihood
  - [ ] modulus
- nodes
  - [x] leaf unobserved variable
- state evolution
  - [x] channels
  - [x] priors
  - [x] likelihood
  - [x] EP algo
- ensemble
  - [ ] Haar measure
  - [ ] gaussian iid
  - [ ] features generator
- GPs
  - [ ] GP prior
  - [ ] sampling operator channel (function -> vector)
  - [ ] linear operator channel (eg derivative, fourier)
- mutual info / evidence
  - [ ] Reeves formula
  - [ ] EP approximation of evidence

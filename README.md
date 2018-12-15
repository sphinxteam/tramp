# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] abs proba_beliefs not normalized
- [x] tau in state evolution

## TODO

- tests
  - [x] channel posterior
  - [x] proba belief normalized
  - [x] second_moment
  - [x] priors
  - [ ] likelihoods
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
  - [ ] proba_beliefs concat channel
  - [ ] proba_beliefs duplicate channel
  - [ ] modulus
  - [ ] ReLU
  - [ ] pow(x, 2)
  - [ ] tanh
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

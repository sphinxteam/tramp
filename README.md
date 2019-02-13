# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] modulus posterior : why numerical integration less precise ?


## TODO

- tests
  - [ ] linear channels (linear, conv, dft, sum, ...)
  - [ ] concat_channel, duplicate_channel
  - [ ] check complex and n-array variables
  - [ ] check second moment computation for non zero mean signal
- model.sample() for variable with n_prev > 1 factors (eg TV/sparse gradient)
- channels
  - [ ] matrix factorization using MAP ?
  - [ ] low matrix factorization (eg learn weights of conv channel) ?
  - [ ] pow(x, n)
- likelihoods
  - [ ] Poisson (eg for photon limited imaging)
- ensembles
  - [ ] features generator
- parameters estimation : view prior as a channel P(sample | parameters)
  - [ ] channel Gaussian(x | r, v)
  - [ ] channel Binary(x | p_pos)
  - [ ] channel GaussBernouilli(x | r, v, rho)
- GPs
  - [ ] GP prior
  - [ ] sampling operator channel (function -> vector)
  - [ ] linear operator channel (eg derivative, fourier)
- mutual info / evidence
  - [ ] I(x,y)
  - [ ] Reeves formula
  - [ ] EP approximation of evidence
- explainer
  - [ ] animation of message passing (store daft every step)
  - [ ] evolution of beliefs / state evo

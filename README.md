# TRAMP

Implement gaussian expectation propagation (aka VAMP)
for any tree-like probabilistic graphical models.
If this package ever achieves a certain level of awesomeness it
will be renamed supertramp.

## FIXME

- [ ] ax=0 in L1 and L12 map priors
- [ ] sng committee : EP diverges
- [ ] mse_ep seems off for gradient channel and conv channels
- [ ] deconv with sparse grad

## TODO

- tests
  - [ ] linear channels (linear, conv, dft, sum, ...)
  - [ ] concat_channel, duplicate_channel
  - [ ] check complex and n-array variables
- model.sample() for variable with n_prev > 1 factors (eg TV/sparse gradient)
- compute_output_shape for each module
- channels
  - [ ] matrix factorization using MAP ?
  - [ ] low rank matrix factorization (eg learn weights of conv channel) ?
  - [ ] x = pow(z, n)
  - [ ] x = exp(alpha z) to model scale variables, eg XRay imaging:
    - signal s is log-density, ie density = 10^s = exp(ln10 s)
    - measurements is intensity = exp(- R) where R = ray integral of density
- likelihoods
  - [ ] Poisson (eg for photon limited imaging)
- models (depend on matrix factorization module)
  - [ ] sparse coding
  - [ ] NMF
  - [ ] conv net
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

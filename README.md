# Tree approximate message passing (TRAMP)

Implements gaussian expectation propagation for any tree-like probabilistic graphical model. 

## Requirements

- numpy/pandas/scipy/matplotlib
- networkx==1.11
- daft

**Warning** Currently the package does not support networkx 2.xx and will throw unexpected errors. We plan to upgrade to networkx 2.xx at some point.

## Examples

Illustrating notebooks and scripts are gathered in the [tramp_notebooks](https://github.com/sphinxteam/tramp_notebooks) repo. 

An illustration of TRAMP on real data-set (MNIST, Fashion-MNIST) for simple inverse problems tasks (inpainting and denoising) is available in [this repo](https://github.com/benjaminaubin/tramp_demo_vae).

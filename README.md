# Tree approximate message passing (TRAMP)

Implements gaussian expectation propagation for any tree-like probabilistic graphical model.

## Requirements

- python>=3.6
- numpy/pandas/scipy/matplotlib
- networkx==1.11
- daft

**Warning** Currently the package does not support networkx 2.xx and will throw unexpected errors. We plan to upgrade to networkx 2.xx at some point.

## Install


To install the package, go to the folder where `setup.py` is located and run:

```
pip install .
```

or if you want to install in development mode (changes to the repository will immediately affect the installed package without needing to re-install):
```
pip install --editable .
```

See [installing from sources](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source) for more details.
In both cases, the necessary requirements should be automatically installed.

## Examples

Illustrating notebooks and scripts are gathered in the [tramp_notebooks](https://github.com/sphinxteam/tramp_notebooks) repo.

An illustration of TRAMP on real data-set (MNIST, Fashion-MNIST) for simple inverse problems tasks (inpainting and denoising) is available in [this repo](https://github.com/benjaminaubin/tramp_demo_vae).

## Acknowledgments

The SPHINX team acknowledges funding from:

![](logos.png)

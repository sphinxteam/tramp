# Tree approximate message passing (Tree-AMP)

Implements gaussian expectation propagation for any tree-like probabilistic graphical model.

Documentation website: [https://sphinxteam.github.io/tramp.docs](https://sphinxteam.github.io/tramp.docs)

## Requirements

- python>=3.6
- numpy/pandas/scipy/matplotlib
- networkx==1.11
- daft

**Warning** Currently the package does not support networkx 2.xx and will throw errors. We plan to upgrade to networkx 2.xx at some point.

## Install

To install the package, go to the folder where `setup.py` is located and run:

```
pip install .
```

or if you want to install in development mode (changes to the repository will immediately affect the installed package without needing to re-install):
```
pip install --editable .
```

To install the package on a remote machine directly from the github repo:
```
pip install git+https://github.com/sphinxteam/tramp.git
```

See [installing from sources](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source) for more details.
In both cases, the necessary requirements should be automatically installed.

## Citation

The package is presented in this [article](https://jmlr.org/papers/v24/20-695.html).
To cite this work:

```
@article{baker2023tree,
  title={Tree-AMP: Compositional inference with tree approximate message passing},
  author={Baker, Antoine and Krzakala, Florent and Aubin, Benjamin and Zdeborová, Lenka},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={57},
  pages={1--89},
  year={2023}
}
```

## Examples

See the corresponding [gallery](https://sphinxteam.github.io/tramp.docs/0.1/html/gallery/index.html) in the documentation website.

## Acknowledgments

Both the SPHINX team and the SMILE team acknowledge funding from:

![](logos.png)

from .gaussian_prior import GaussianPrior
from .gauss_bernouilli_prior import GaussBernouilliPrior
from .binary_prior import BinaryPrior
from .map_laplace_prior import MAP_LaplacePrior
from .map_L21_norm_prior import MAP_L21NormPrior

PRIOR_CLASSES = {
    "gaussian": GaussianPrior,
    "gauss_bernouilli": GaussBernouilliPrior,
    "binary": BinaryPrior,
    "laplace": MAP_LaplacePrior,
    "L21_norm": MAP_L21NormPrior
}


def get_prior(size, prior_type, **kwargs):
    prior = PRIOR_CLASSES[prior_type](size=size, **kwargs)
    return prior

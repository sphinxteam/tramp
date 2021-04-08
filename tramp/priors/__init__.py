from .gaussian_prior import GaussianPrior
from .gauss_bernoulli_prior import GaussBernoulliPrior
from .binary_prior import BinaryPrior
from .map_laplace_prior import MAP_LaplacePrior
from .map_L21_norm_prior import MAP_L21NormPrior
from .exponential_prior import ExponentialPrior

PRIOR_CLASSES = {
    "gaussian": GaussianPrior,
    "gauss_bernoulli": GaussBernoulliPrior,
    "binary": BinaryPrior,
    "laplace": MAP_LaplacePrior,
    "L21_norm": MAP_L21NormPrior,
    "exponential": ExponentialPrior
}


def get_prior(size, prior_type, **kwargs):
    prior = PRIOR_CLASSES[prior_type](size=size, **kwargs)
    return prior

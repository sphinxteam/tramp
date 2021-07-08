"""Gathers the priors."""
from .gaussian_prior import GaussianPrior
from .gauss_bernoulli_prior import GaussBernoulliPrior
from .binary_prior import BinaryPrior
from .map_L1_norm_prior import MAP_L1NormPrior
from .map_L21_norm_prior import MAP_L21NormPrior
from .exponential_prior import ExponentialPrior
from .positive_prior import PositivePrior
from .gaussian_mixture_prior import GaussianMixturePrior
from .committee_binary_prior import CommitteeBinaryPrior

PRIOR_CLASSES = {
    "gaussian": GaussianPrior,
    "gauss_bernoulli": GaussBernoulliPrior,
    "binary": BinaryPrior,
    "L1_norm": MAP_L1NormPrior,
    "L21_norm": MAP_L21NormPrior,
    "exponential": ExponentialPrior,
    "positive": ExponentialPrior,
    "mixture": GaussianMixturePrior,
    "committee_binary": CommitteeBinaryPrior
}


def get_prior(size, prior_type, **kwargs):
    prior = PRIOR_CLASSES[prior_type](size=size, **kwargs)
    return prior

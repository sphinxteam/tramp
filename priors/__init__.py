from .gaussian_prior import GaussianPrior
from .gauss_bernouilli_prior import GaussBernouilliPrior
from .binary_prior import BinaryPrior

PRIOR_CLASSES = {
    "gaussian": GaussianPrior,
    "gauss_bernouilli": GaussBernouilliPrior,
    "binary": BinaryPrior
}

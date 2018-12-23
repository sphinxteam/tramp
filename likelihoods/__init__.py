from .gaussian_likelihood import GaussianLikelihood
from .sng_likelihood import SngLikelihood
from .abs_likelihood import AbsLikelihood
from .modulus_likelihood import ModulusLikelihood


LIKELIHOOD_CLASSES = {
    "gaussian": GaussianLikelihood,
    "abs": AbsLikelihood,
    "sng": SngLikelihood,
    "modulus": ModulusLikelihood
}

from .gaussian_likelihood import GaussianLikelihood
from .sgn_likelihood import SgnLikelihood
from .abs_likelihood import AbsLikelihood
from .modulus_likelihood import ModulusLikelihood


LIKELIHOOD_CLASSES = {
    "gaussian": GaussianLikelihood,
    "abs": AbsLikelihood,
    "sgn": SgnLikelihood,
    "modulus": ModulusLikelihood
}

def get_likelihood(y, likelihood_type):
    likelihood = LIKELIHOOD_CLASSES[likelihood_type](y=y)
    return likelihood

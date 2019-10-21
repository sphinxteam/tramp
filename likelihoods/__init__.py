from .gaussian_likelihood import GaussianLikelihood
from .sgn_likelihood import SgnLikelihood
from .abs_likelihood import AbsLikelihood
from .modulus_likelihood import ModulusLikelihood
from .piecewise_linear_likelihood import (
    ReluLikelihood, LeakyReluLikelihood, AsymmetricAbsLikelihood,
    HardTanhLikelihood, HardSigmoidLikelihood, SymmetricDoorLikelihood
)


LIKELIHOOD_CLASSES = {
    "gaussian": GaussianLikelihood,
    "abs": AbsLikelihood,
    "sgn": SgnLikelihood,
    "door": SymmetricDoorLikelihood,
    "relu": ReluLikelihood,
    "l-relu": LeakyReluLikelihood,
    "h-tanh": HardTanhLikelihood,
    "h-sigm": HardSigmoidLikelihood,
    "a-abs": AsymmetricAbsLikelihood,
    "modulus": ModulusLikelihood
}


def get_likelihood(y, likelihood_type, **kwargs):
    likelihood = LIKELIHOOD_CLASSES[likelihood_type](y=y, **kwargs)
    return likelihood

from .gaussian_ensemble import GaussianEnsemble
from .complex_gaussian_ensemble import ComplexGaussianEnsemble
from .rotation_ensemble import RotationEnsemble
from .unitary_ensemble import UnitaryEnsemble
from .binary_ensemble import BinaryEnsemble
from .ternary_ensemble import TernaryEnsemble
from .marchenko_pastur_ensemble import MarchenkoPasturEnsemble

ENSEMBLE_CLASSES = {
    "gaussian": GaussianEnsemble,
    "complex_gaussian":ComplexGaussianEnsemble,
    "rotation": RotationEnsemble,
    "unitary": UnitaryEnsemble,
    "binary": BinaryEnsemble,
    "ternary": TernaryEnsemble,
    "marchenko": MarchenkoPasturEnsemble
}


def get_ensemble(ensemble_type, **kwargs):
    ensemble = ENSEMBLE_CLASSES[ensemble_type](**kwargs)
    return ensemble

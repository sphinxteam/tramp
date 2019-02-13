from .gaussian_ensemble import GaussianEnsemble
from .rotation_ensemble import RotationEnsemble
from .unitary_ensemble import UnitaryEnsemble
from .binary_ensemble import BinaryEnsemble
from .ternary_ensemble import TernaryEnsemble


ENSEMBLE_CLASSES = {
    "gaussian": GaussianEnsemble,
    "rotation": RotationEnsemble,
    "unitary": UnitaryEnsemble,
    "binary": BinaryEnsemble,
    "ternary": TernaryEnsemble
}


def get_ensemble(ensemble_type, **kwargs):
    ensemble_kwargs = dict(N=kwargs["N"])
    if ensemble_type in ["gaussian", "binary"]:
        ensemble_kwargs["M"] = kwargs["M"]
    if ensemble_type == "binary":
        ensemble_kwargs["p_pos"] = kwargs["p_pos"]
    if ensemble_type == "ternary":
        ensemble_kwargs["p_pos"] = kwargs["p_pos"]
        ensemble_kwargs["p_neg"] = kwargs["p_neg"]
    ensemble = ENSEMBLE_CLASSES[ensemble_type](**ensemble_kwargs)
    return ensemble

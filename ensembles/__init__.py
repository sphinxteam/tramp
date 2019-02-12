from .gaussian_ensemble import GaussianEnsemble
from .rotation_ensemble import RotationEnsemble
from .unitary_ensemble import UnitaryEnsemble


ENSEMBLE_CLASSES = {
    "gaussian": GaussianEnsemble,
    "rotation": RotationEnsemble,
    "unitary": UnitaryEnsemble
}

def get_ensemble(ensemble_type, **kwargs):
    ensemble_kwargs = dict(N=kwargs["N"])
    if ensemble_type in ["gaussian"]:
        ensemble_kwargs["M"] = kwargs["M"]
    ensemble = ENSEMBLE_CLASSES[ensemble_type](**ensemble_kwargs)
    return ensemble

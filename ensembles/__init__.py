from .gaussian_ensemble import GaussianEnsemble

ENSEMBLE_CLASSES = {
    "gaussian": GaussianEnsemble
}

def get_ensemble(ensemble_type, **kwargs):
    ensemble_kwargs = dict(M=kwargs["M"] , N=kwargs["N"])
    prior = ENSEMBLE_CLASSES[ensemble_type](**ensemble_kwargs)
    return prior

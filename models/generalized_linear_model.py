from ..channels import get_channel
from ..priors import get_prior
from ..ensembles import get_ensemble
from ..likelihoods import get_likelihood
from ..variables import SISOVariable as V, SILeafVariable as O


def get_kwargs(target, kwargs):
    n_char = len(target) + 1
    target_kwargs = {
        key[n_char:]:val for key, val in kwargs.items()
        if key.startswith(target)
    }
    return target_kwargs


def glm_generative(N, alpha, ensemble_type, prior_type, output_type, **kwargs):
    "Build a generative Generalized Linear Model"
    # sensing matrix
    M = int(alpha * N)
    ensemble_kwargs = get_kwargs("ensemble", kwargs)
    ensemble = get_ensemble(ensemble_type, M=M, N=N, **ensemble_kwargs)
    F = ensemble.generate()
    # factors
    prior_kwargs = get_kwargs("prior", kwargs)
    size = (2, N) if output_type=="modulus" else N
    prior = get_prior(size=size, prior_type=prior_type, **prior_kwargs)
    linear_type = "complex_linear" if output_type=="modulus" else "linear"
    linear = get_channel(linear_type, W=F, name="F")
    output_kwargs = get_kwargs("output", kwargs)
    output = get_channel(channel_type=output_type, **output_kwargs)
    # model
    model = (
        prior @ V(id="x") @ linear @ V(id="z") @ output @ O(id="y")
    ).to_model()
    return model

def glm_state_evolution(alpha, prior_type, output_type, **kwargs):
    """
    Build a Generalized Linear Model to be used only for State Evolution. The
    linear channels are Marchenko-Pastur.
    """
    # factors
    prior_kwargs = get_kwargs("prior", kwargs)
    prior = get_prior(size=None, prior_type=prior_type, **prior_kwargs)
    linear = get_channel("marchenko", alpha=alpha, name="F")
    output_kwargs = get_kwargs("output", kwargs)
    output = get_likelihood(
        y=None, y_name="y", likelihood_type=output_type, **output_kwargs
    )
    # model
    model = (
        prior @ V(id="x") @ linear @ V(id="z") @ output
    ).to_model()
    return model

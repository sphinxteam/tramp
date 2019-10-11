from ..variables import SISOVariable as V, SILeafVariable as O
from ..priors import get_prior
from ..channels import get_channel, SumChannel, LinearChannel, GaussianChannel
from ..ensembles import get_ensemble


def committee(K, N, alpha, ensemble_type, priors, activation1, activation2,
              noise_var):
    if activation1 not in ["abs", "relu", "sgn"]:
        raise ValueError(
            f"activation1={activation1} must be abs, sgn or relu"
        )
    if activation2 not in [None, "abs", "relu", "sgn"]:
        raise ValueError(
            f"activation2={activation2} must be None, abs, sgn or relu"
        )
    if not isinstance(priors, list) or len(priors) != K:
        raise ValueError(f"priors must be a list of length {K}")
    # sensing matrix
    M = int(alpha * N)
    ensemble = get_ensemble(ensemble_type, M=M, N=N)
    F = ensemble.generate()
    # K experts
    experts = None
    for k, prior in enumerate(priors):
        expert = (
            get_prior(size=N, **prior) @
            V(id=f"x_{k}") @
            LinearChannel(F, name="F") @
            V(id=f"z_{k}") @
            get_channel(activation1) @
            V(id=f"a_{k}")
        )
        experts = expert if experts is None else experts + expert
    # committee of the K experts
    model_dag = experts @ SumChannel(n_prev=K)
    if activation2 in ["abs", "relu", "sgn"]:
        model_dag = model_dag @ V(id="a") @ get_channel(activation2)
    if noise_var:
        model_dag = model_dag @ V(id="n") @ GaussianChannel(var=noise_var)
    model_dag = model_dag @ O(id="y")
    return model_dag.to_model()


def sgn_committee(K, N, alpha, ensemble_type, p_pos, noise_var):
    if isinstance(p_pos, float):
        p_pos = [p_pos]*K
    if not isinstance(p_pos, list) or len(p_pos) != K:
        raise ValueError(f"p_pos must be a list of length {K}")
    priors = [dict(prior_type="binary", p_pos=p) for p in p_pos]
    activation1 = activation2 = "sgn"
    return committee(
        K, N, alpha, ensemble_type, priors, activation1, activation2, noise_var
    )


def soft_committee(K, N, alpha, ensemble_type, prior_mean, prior_var, noise_var):
    if isinstance(prior_mean, float):
        prior_mean = [prior_mean]*K
    if not isinstance(prior_mean, list) or len(prior_mean) != K:
        raise ValueError(f"prior_mean must be a list of length {K}")
    if isinstance(prior_var, float):
        prior_var = [prior_var]*K
    if not isinstance(prior_var, list) or len(prior_var) != K:
        raise ValueError(f"prior_var must be a list of length {K}")
    priors = [
        dict(prior_type="gaussian", mean=m, var=v)
        for m, v in zip(prior_mean, prior_var)
    ]
    activation1, activation2 = "relu", None
    return committee(
        K, N, alpha, ensemble_type, priors, activation1, activation2, noise_var
    )

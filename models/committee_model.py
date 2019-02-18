from ..base import SISOVariable, SILeafVariable
from .dag_model import DAGModel
from ..priors import get_prior
from ..channels import get_channel, SumChannel, LinearChannel, GaussianChannel
from ..ensembles import get_ensemble


class Committee(DAGModel):

    def __init__(self, K, N, alpha, ensemble_type,
                 priors, activation1, activation2, var_noise=None):
        if activation1 not in ["abs", "relu", "sng"]:
            raise ValueError(
                f"activation1={activation1} must be abs, sng or relu"
            )
        if activation2 not in ["identity", "abs", "relu", "sng"]:
            raise ValueError(
                f"activation2={activation2} must be identity, abs, sng or relu"
            )
        if not isinstance(priors, list) or len(priors) != K:
            raise ValueError(f"priors must be a list of length {K}")
        # sensing matrix
        M = int(alpha * N)
        self.ensemble = get_ensemble(ensemble_type, M=M, N=N)
        F = self.ensemble.generate()
        # model
        self.K = K
        self.priors = priors
        self.activation1 = activation1
        self.activation2 = activation2
        self.repr_init(pad="  ")
        # K experts
        experts = None
        for k, prior in enumerate(priors):
            expert = (
                get_prior(size=N, **prior) @
                SISOVariable(id=f"x_{k}") @
                LinearChannel(F, W_name="F") @
                SISOVariable(id=f"z_{k}") @
                get_channel(activation1) @
                SISOVariable(id=f"a_{k}")
            )
            experts = expert if experts is None else experts + expert
        # committee of the K experts
        model_dag = experts @ SumChannel(n_prev=K)
        if activation2 in ["abs", "relu", "sng"]:
            model_dag = model_dag @ SISOVariable(id="a") @ get_channel(activation2)
        if var_noise:
            model_dag = model_dag @ SISOVariable(id="n") @ GaussianChannel(var=var_noise)
        model_dag = model_dag @ SILeafVariable(id="y")
        model_dag = model_dag.to_model_dag()
        DAGModel.__init__(self, model_dag)


class SngCommittee(Committee):
    def __init__(self, K, N, alpha, ensemble_type,
                 p_pos, var_noise=None):
        if not isinstance(p_pos, list) or len(p_pos) != K:
            raise ValueError(f"p_pos must be a list of length {K}")
        priors = [
            dict(prior_type="binary", p_pos=p)
            for p in p_pos
        ]
        activation1 = "sng"
        activation2 = "sng"
        super().__init__(
            K, N, alpha,
            ensemble_type, priors, activation1, activation2,
            var_noise
        )


class SoftCommittee(Committee):
    def __init__(self, K, N, alpha, ensemble_type,
                 mean_prior, var_prior, var_noise=None):
        if not isinstance(mean_prior, list) or len(mean_prior) != K:
            raise ValueError(f"mean_prior must be a list of length {K}")
        if not isinstance(var_prior, list) or len(var_prior) != K:
            raise ValueError(f"var_prior must be a list of length {K}")
        priors = [
            dict(prior_type="gaussian", mean_prior=m, var_prior=v)
            for m, v in zip(mean_prior, var_prior)
        ]
        activation1 = "relu"
        activation2 = "identity"
        super().__init__(
            K, N, alpha,
            ensemble_type, priors, activation1, activation2,
            var_noise
        )

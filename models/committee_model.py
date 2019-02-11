from ..base import SISOVariable, SIMOVariable, MISOVariable, SILeafVariable
from .dag_model import DAGModel
from ..priors import get_prior
from ..channels import get_channel, SumChannel, LinearChannel, GaussianChannel
from ..ensembles import get_ensemble

class Committee(DAGModel):

    def __init__(self, K, N, alpha,
                 ensemble_type, prior_type, activation1, activation2,
                 **kwargs):
        if activation1 not in ["abs", "relu", "sng"]:
            raise ValueError(
                f"activation1={activation1} must be abs, sng or relu"
            )
        if activation2 not in ["identity", "abs", "relu", "sng"]:
            raise ValueError(
                f"activation2={activation2} must be identity, abs, sng or relu"
            )
        # sensing matrix
        M = int(alpha * N)
        self.ensemble = get_ensemble(ensemble_type, M=M, N=N)
        F = self.ensemble.generate()
        # model
        self.K = K
        self.activation1 = activation1
        self.activation2 = activation2
        self.repr_init(pad="  ")
        # K experts
        experts = None
        for k in range(K):
            expert = (
                get_prior(N, prior_type, **kwargs) @
                SISOVariable(id=f"x_{k}") @
                LinearChannel(F, W_name="F") @
                SISOVariable(id=f"z_{k}") @
                get_channel(activation1) @
                SISOVariable(id=f"a_{k}")
            )
            experts = expert if experts is None else experts + expert
        # committee of the K experts
        model_dag = experts @ SumChannel(n_prev=K)
        if "var_noise" in kwargs:
            model_dag = model_dag @ GaussianChannel(var=var_noise)
        if activation2 in ["abs", "relu", "sng"]:
            model_dag = model_dag @ SISOVariable(id="a") @ get_channel(activation2)
        model_dag = model_dag @ SILeafVariable(id="y")
        model_dag = model_dag.to_model_dag()
        DAGModel.__init__(self, model_dag)

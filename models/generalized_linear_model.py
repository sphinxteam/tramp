from ..base import ReprMixin
from .multi_layer_model import MultiLayerModel
from ..channels import get_channel, GaussianChannel, LinearChannel
from ..priors import get_prior
from ..ensembles import get_ensemble


class GaussianDenoiser(ReprMixin, MultiLayerModel):
    def __init__(self, N, prior_type, var_noise, **kwargs):
        self.prior = get_prior(N, prior_type, **kwargs)
        self.channel = GaussianChannel(var=var_noise)
        self.repr_init(pad="  ")
        MultiLayerModel.__init__(self, [self.prior, self.channel])


class GeneralizedLinearModel(ReprMixin, MultiLayerModel):
    def __init__(self, N, alpha, ensemble_type, prior_type, output_type, **kwargs):
        # sensing matrix
        M = int(alpha * N)
        self.ensemble = get_ensemble(ensemble_type, **kwargs)
        F = self.ensemble.generate(M, N)
        # model
        self.prior = get_prior(N, prior_type, **kwargs)
        self.linear = LinearChannel(F)
        self.output = get_channel(output_type, **kwargs)
        self.repr_init(pad="  ")
        MultiLayerModel.__init__(self, [self.prior, self.linear, self.output])

class SparseRegression(GeneralizedLinearModel):
    def __init__(self, N, alpha, ensemble_type, rho, var_noise):
        prior_type = "gauss_bernouilli"
        output_type = "gaussian"
        super().__init__(
            N, alpha, ensemble_type, prior_type, output_type,
            rho=rho, var_noise=var_noise
        )

class RidgeRegression(GeneralizedLinearModel):
    def __init__(self, N, alpha, ensemble_type, var_noise):
        prior_type = "gaussian"
        output_type = "gaussian"
        super().__init__(
            N, alpha, ensemble_type, prior_type, output_type,
            var_noise=var_noise
        )

class Perceptron(GeneralizedLinearModel):
    def __init__(self, N, alpha, ensemble_type, p_pos):
        prior_type = "binary"
        output_type = "sng"
        super().__init__(
            N, alpha, ensemble_type, prior_type, output_type,
            p_pos=p_pos
        )

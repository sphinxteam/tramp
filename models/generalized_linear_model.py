from ..base import ReprMixin
from .multi_layer_model import MultiLayerModel
from ..channels import CHANNEL_CLASSES, GaussianChannel, LinearChannel
from ..priors import PRIOR_CLASSES
from ..ensembles import ENSEMBLE_CLASSES


def get_prior(size, prior_type, **kwargs):
    prior_kwargs = {}
    if prior_type=="binary":
        prior_kwargs["p_pos"]=kwargs["p_pos"]
    if prior_type=="gauss_bernouilli":
        prior_kwargs["rho"]=kwargs["rho"]
    prior = PRIOR_CLASSES[prior_type](size=size, **prior_kwargs)
    return prior

def get_channel(channel_type, **kwargs):
    channel_kwargs = {}
    if channel_type=="gaussian":
        channel_kwargs["var"]=kwargs["var_noise"]
    channel = CHANNEL_CLASSES[channel_type](**channel_kwargs)
    return channel


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
        self.ensemble = ENSEMBLE_CLASSES[ensemble_type]()
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

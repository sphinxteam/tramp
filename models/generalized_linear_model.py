from .multi_layer_model import MultiLayerModel
from ..channels import (
    get_channel, GaussianChannel, LinearChannel,
    ComplexLinearChannel, ModulusChannel
)
from ..priors import get_prior
from ..ensembles import get_ensemble


class GaussianDenoiser(MultiLayerModel):
    def __init__(self, N, prior_type, var_noise, **kwargs):
        self.prior = get_prior(N, prior_type, **kwargs)
        self.channel = GaussianChannel(var=var_noise)
        self.repr_init(pad="  ")
        super().__init__(layers=[self.prior, self.channel])


class GeneralizedLinearModel(MultiLayerModel):
    def __init__(self, N, alpha, ensemble_type, prior_type, output_type, **kwargs):
        # sensing matrix
        M = int(alpha * N)
        self.ensemble = get_ensemble(ensemble_type, M=M, N=N)
        F = self.ensemble.generate()
        # model
        self.prior = get_prior(N, prior_type, **kwargs)
        self.linear = LinearChannel(F, W_name="F")
        self.output = get_channel(output_type, **kwargs)
        self.repr_init(pad="  ")
        super().__init__(
            layers=[self.prior, self.linear, self.output],
            ids=["x", "z", "y"]
        )

class PhaseRetrieval(MultiLayerModel):
    def __init__(self, N, alpha, ensemble_type, prior_type, **kwargs):
        # sensing matrix
        M = int(alpha * N)
        self.ensemble = get_ensemble(ensemble_type, M=M, N=N)
        F = self.ensemble.generate()
        # model
        x_size = (2, N) # complex signal x
        self.prior = get_prior(x_size, prior_type, **kwargs)
        self.linear = ComplexLinearChannel(F, W_name="F")
        self.output = ModulusChannel()
        self.repr_init(pad="  ")
        super().__init__(
            layers=[self.prior, self.linear, self.output],
            ids=["x", "z", "y"]
        )


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

class SgnRetrieval(GeneralizedLinearModel):
    def __init__(self, N, alpha, ensemble_type, mean_prior):
        prior_type = "gaussian"
        output_type = "abs"
        super().__init__(
            N, alpha, ensemble_type, prior_type, output_type,
            mean_prior=mean_prior
        )


class Perceptron(GeneralizedLinearModel):
    def __init__(self, N, alpha, ensemble_type, p_pos):
        prior_type = "binary"
        output_type = "sgn"
        super().__init__(
            N, alpha, ensemble_type, prior_type, output_type,
            p_pos=p_pos
        )

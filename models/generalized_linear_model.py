from ..base import ReprMixin
from .multi_layer_model import MultiLayerModel
from ..channels import LinearChannel

class GeneralizedLinearModel(MultiLayerModel, ReprMixin):
    def __init__(self, prior, channel, likelihood):
        if not isinstance(channel, LinearChannel):
            raise ValueError(f"channel must be linear: got {channel}")
        self.prior = prior
        self.channel = channel
        self.likelihood = likelihood
        self.repr_init(pad="\t")
        layers = [prior, channel, likelihood]
        MultiLayerModel.__init__(self, layers)

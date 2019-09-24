# noise
from .noise.gaussian_channel import GaussianChannel
# shape
from .shape.concat_channel import ConcatChannel
from .shape.duplicate_channel import DuplicateChannel
from .shape.reshape_channel import ReshapeChannel
# linear
from .linear.sum_channel import SumChannel
from .linear.dft_channel import DFTChannel
from .linear.bias_channel import BiasChannel
from .linear.rotation_channel import RotationChannel
from .linear.unitary_channel import UnitaryChannel
from .linear.linear_channel import LinearChannel
from .linear.complex_linear_channel import ComplexLinearChannel
from .linear.conv_channel import (
    ConvChannel, Blur1DChannel, Blur2DChannel,
    DifferentialChannel, LaplacianChannel
)
from .linear.gradient_channel import GradientChannel
from .linear.analytical_linear_channel import (
    AnalyticalLinearChannel, MarchenkoPasturChannel
)
# activation
from .activation.sgn_channel import SgnChannel
from .activation.abs_channel import AbsChannel
from .activation.relu_channel import ReluChannel
from .activation.leaky_relu_channel import LeakyReluChannel
from .activation.hard_tanh_channel import HardTanhChannel
from .activation.tanh_channel import TanhChannel
from .activation.modulus_channel import ModulusChannel
# low rank
from .low_rank.low_rank_gram_channel import LowRankGramChannel
from .low_rank.low_rank_factorization import LowRankFactorization


CHANNEL_CLASSES = {
    "sum": SumChannel,
    "dft": DFTChannel,
    "concat": ConcatChannel,
    "duplicate": DuplicateChannel,
    "gaussian": GaussianChannel,
    "rotation": RotationChannel,
    "unitary": UnitaryChannel,
    "linear": LinearChannel,
    "complex_linear": ComplexLinearChannel,
    "conv": ConvChannel,
    "blur_1d": Blur1DChannel,
    "blur_2d": Blur2DChannel,
    "diff": DifferentialChannel,
    "laplacian": LaplacianChannel,
    "gradient": GradientChannel,
    "sgn": SgnChannel,
    "abs": AbsChannel,
    "relu": ReluChannel
}


def get_channel(channel_type, **kwargs):
    channel_kwargs = {}
    if channel_type == "gaussian":
        channel_kwargs["var"] = kwargs["var_noise"]
    channel = CHANNEL_CLASSES[channel_type](**channel_kwargs)
    return channel

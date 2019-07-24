from .sum_channel import SumChannel
from .dft_channel import DFTChannel
from .bias_channel import BiasChannel
from .concat_channel import ConcatChannel
from .duplicate_channel import DuplicateChannel
from .gaussian_channel import GaussianChannel
from .rotation_channel import RotationChannel
from .unitary_channel import UnitaryChannel
from .linear_channel import LinearChannel
from .complex_linear_channel import ComplexLinearChannel
from .conv_channel import (
    ConvChannel, Blur1DChannel, Blur2DChannel,
    DifferentialChannel, LaplacianChannel
)
from .gradient_channel import GradientChannel
from .sgn_channel import SgnChannel
from .abs_channel import AbsChannel
from .relu_channel import ReluChannel
from .leaky_relu_channel import LeakyReluChannel
from .hard_tanh_channel import HardTanhChannel
from .tanh_channel import TanhChannel
from .modulus_channel import ModulusChannel
from .reshape_channel import ReshapeChannel
from .low_rank_gram_channel import LowRankGramChannel
from .low_rank_factorization import LowRankFactorization


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

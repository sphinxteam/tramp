from .sum_channel import SumChannel
from .dft_channel import DFTChannel
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
from .sng_channel import SngChannel
from .abs_channel import AbsChannel
from .relu_channel import ReluChannel
from .modulus_channel import ModulusChannel


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
    "sng": SngChannel,
    "abs": AbsChannel,
    "relu": ReluChannel
}


def get_channel(channel_type, **kwargs):
    channel_kwargs = {}
    if channel_type == "gaussian":
        channel_kwargs["var"] = kwargs["var_noise"]
    channel = CHANNEL_CLASSES[channel_type](**channel_kwargs)
    return channel

from .sum_channel import SumChannel
from .concat_channel import ConcatChannel
from .duplicate_channel import DuplicateChannel
from .gaussian_channel import GaussianChannel
from .rotation_channel import RotationChannel
from .linear_channel import LinearChannel
from .conv_channel import ConvChannel
from .gradient_channel import GradientChannel
from .sng_channel import SngChannel
from .abs_channel import AbsChannel
from .relu_channel import ReluChannel

CHANNEL_CLASSES = {
    "sum": SumChannel,
    "concat": ConcatChannel,
    "duplicate": DuplicateChannel,
    "gaussian": GaussianChannel,
    "rotation": RotationChannel,
    "linear": LinearChannel,
    "conv": ConvChannel,
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

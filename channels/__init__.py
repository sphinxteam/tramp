from .concat_channel import ConcatChannel
from .duplicate_channel import DuplicateChannel
from .gaussian_channel import GaussianChannel
from .rotation_channel import RotationChannel
from .linear_channel import LinearChannel
from .sng_channel import SngChannel
from .abs_channel import AbsChannel

CHANNEL_CLASSES = {
    "concat": ConcatChannel,
    "duplicate": DuplicateChannel,
    "gaussian": GaussianChannel,
    "rotation": RotationChannel,
    "linear": LinearChannel,
    "sng": SngChannel,
    "abs": AbsChannel
}


def get_channel(channel_type, **kwargs):
    channel_kwargs = {}
    if channel_type == "gaussian":
        channel_kwargs["var"] = kwargs["var_noise"]
    channel = CHANNEL_CLASSES[channel_type](**channel_kwargs)
    return channel

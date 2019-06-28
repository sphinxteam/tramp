import numpy as np
from .activation_channel import ActivationChannel

class TanhChannel(ActivationChannel):
    def __init__(self):
        super().__init__(func = np.tanh)

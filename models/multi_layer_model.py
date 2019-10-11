from ..priors.base_prior import Prior
from ..channels.base_channel import Channel
from ..likelihoods.base_likelihood import Likelihood
from ..variables import SISOVariable, SILeafVariable
from .base_model import Model


def check_layers(layers):
    if not isinstance(layers[0], Prior):
        raise ValueError("first layer must be a Prior")
    for i, layer in enumerate(layers[1:-1]):
        if not isinstance(layer, Channel):
            raise ValueError(f"intermediate layer i={i} must be a Channel")
    if isinstance(layers[-1], Channel):
        if layers[-1].n_next != 1:
            raise ValueError("last layer must be a Channel with one output")
    elif not isinstance(layers[-1], Likelihood):
        raise ValueError("last layer must be a Channel or a Likelihood")


def default_ids(n_layers):
    "Return x, t_1, ..., t_{L-1}, y"
    ids = [f"t_{l}" for l in range(n_layers)]
    ids[0] = "x"
    if n_layers > 1:
        ids[-1] = "y"
    return ids


class MultiLayerModel(Model):
    def __init__(self, layers, ids=None):
        check_layers(layers)
        n_layers = len(layers)
        ids = ids or default_ids(n_layers)
        if len(ids) != n_layers:
            raise ValueError(f"ids should be of length {n_layers}")
        self.n_layers = n_layers
        self.layers = layers
        self.ids = ids
        self.repr_init(pad="  ")

        def get_variable(l):
            V = SILeafVariable if l == n_layers-1 else SISOVariable
            return V(id=ids[l])
        dag = layers[0] @ get_variable(0)
        for l in range(1, n_layers):
            dag = dag @ layers[l] @ get_variable(l)
        model_dag = dag.to_model_dag()
        Model.__init__(self, model_dag)

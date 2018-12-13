from ..base import Prior, Channel, Likelihood
from .factor_pgm import FactorPGM
import networkx as nx


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


class MultiLayerModel(FactorPGM):
    def __init__(self, layers):
        check_layers(layers)
        self.observed = isinstance(layers[-1], Likelihood)
        self.layers = layers
        factor_dag = self._build_factor_dag(layers)
        FactorPGM.__init__(self, factor_dag)

    def __repr__(self):
        pad = "  "
        padded = "\n".join([
            f"{pad}{layer}," for layer in self.layers
        ])
        inner = f"\n{padded}\n{pad}observed={self.observed}\n"
        return f"MultiLayer({inner})"

    def _build_factor_dag(self, layers):
        dag = nx.DiGraph()
        dag.add_nodes_from(layers)
        dag.add_edges_from(zip(layers[:-1], layers[1:]))
        return dag

import networkx as nx
import numpy as np


class Layout():
    def __init__(self, dx=0.7, dy=0.7, margin=1):
        self.dx = dx
        self.dy = dy
        self.margin = margin
        self.dag = None

    def compute_dag(self, dag):
        self._init_dag(dag)
        self.forward_message()
        self._fix_limits()

    def _init_dag(self, dag):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(f"dag {dag} not a DAG")
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(dag.edges())
        self.forward_ordering = nx.topological_sort(dag)
        self.backward_ordering = list(reversed(self.forward_ordering))
        for order, node in enumerate(self.forward_ordering):
            self.dag.node[node].update(order=order)
        root_order = 0
        for node in self.forward_ordering:
            root_node = not self.dag.predecessors(node)
            if root_node:
                self.dag.node[node].update(root_order=root_order)
                root_order += 1

    def _fix_limits(self):
        xmin = min([data["x"] for node, data in self.dag.nodes(data=True)])
        ymin = min([data["y"] for node, data in self.dag.nodes(data=True)])
        for node, data in self.dag.nodes(data=True):
            data["x"] = data["x"] - xmin + self.margin
            data["y"] = data["y"] - ymin + self.margin
        xmax = max([data["x"] for node, data in self.dag.nodes(data=True)])
        ymax = max([data["y"] for node, data in self.dag.nodes(data=True)])
        self.Lx = xmax + self.margin
        self.Ly = ymax + self.margin

    def forward(self, node, message):
        assert "x" not in self.dag.node[node]
        root_node = not self.dag.predecessors(node)
        if root_node:
            x = 0
            y = self.dy * self.dag.node[node]["root_order"]
        else:
            x = np.max([data["x"] for source, target, data in message])
            y = np.mean([data["y"] for source, target, data in message])
        self.dag.node[node].update(x=x, y=y)
        targets = sorted(
            self.dag.successors(node),
            key=lambda n: self.dag.node[n]["order"]
        )
        N = len(targets)
        y_targets = y + self.dy * np.linspace(-0.5 * (N - 1), 0.5 * (N - 1), N)
        new_message = [
            (node, target, dict(x=x + self.dx, y=y_target))
            for y_target, target in zip(y_targets, targets)
        ]
        return new_message

    def update_message(self, new_message):
        for source, target, new_data in new_message:
            self.dag[source][target].update(new_data)

    def forward_message(self):
        for node in self.forward_ordering:
            message = self.dag.in_edges(node, data=True)
            new_message = self.forward(node, message)
            self.update_message(new_message)

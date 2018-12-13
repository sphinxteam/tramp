import networkx as nx

class Layout():
    def __init__(self, dx=0.7, dy=0.7, margin=1):
        self.dx = dx
        self.dy = dy
        self.margin = margin
        self.dag = None

    def compute_dag(self, dag):
        self._check_dag(dag)
        self._init_dag(dag)
        root_node = nx.topological_sort(self.dag)[0]
        start_nodes = [(root_node, 0, 0, +1)]
        self._step(start_nodes)
        self._fix_limits()
        return self

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

    def _step(self, nodes):
        # may cause stack overflow...
        for node, x, y, dir in nodes:
            if self.dag.node[node]["done"]:
                logging.warning(f"step: node {node} already done")
                next_nodes = []
            if self.dag.node[node]["line"]:
                next_nodes = self._handle_line(node, x, y, dir)
            if self.dag.node[node]["join"]:
                next_nodes = self._handle_join(node, x, y, dir)
            if self.dag.node[node]["fork"]:
                next_nodes = self._handle_fork(node, x, y, dir)
            if next_nodes:
                self._step(next_nodes)
        return

    def _check_dag(self, dag):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(f"dag {dag} not a DAG")
        for node in dag.nodes():
            predecessors = dag.predecessors(node)
            successors = dag.successors(node)
            n_prev = len(predecessors)
            n_next = len(successors)
            if (n_prev > 2):
                raise NotImplementedError(f"cannot handle node with {n_prev} predecessors")
            if (n_next > 2):
                raise NotImplementedError(f"cannot handle node with {n_next} successors")

    def _init_dag(self, dag):
        self.dag = nx.DiGraph()
        self.dag.add_edges_from(dag.edges())
        for node in self.dag:
            predecessors = self.dag.predecessors(node)
            successors = self.dag.successors(node)
            n_prev = len(predecessors)
            n_next = len(successors)
            root_node = n_prev == 0
            leaf_node = n_next == 0
            line_node = n_prev <= 1 and n_next <= 1
            fork_node = n_prev <= 1 and n_next == 2
            join_node = n_prev == 2 and n_next <= 1
            self.dag.node[node].update(
                root=root_node, leaf=leaf_node,
                line=line_node, fork=fork_node, join=join_node,
                x=None, y=None, dir=None, done=False
            )

    def _handle_join(self, node, x, y, dir):
        assert self.dag.node[node]["join"]
        y_join = y if (dir == -1) else y + 0.5 * self.dy
        self.dag.node[node].update(x=x, y=y_join, dir=dir, done=True)
        todo_successors = [
            (node, x + self.dx, y_join, +1)
            for node in self.dag.successors(node)
            if not self.dag.node[node]["done"]
        ]
        if dir == +1:
            y_done = [
                self.dag.node[node]["y"]
                for node in self.dag.predecessors(node)
                if self.dag.node[node]["done"]
            ]
            assert len(y_done) == 1
            y_done = y_done[0]
            todo_predecessors = [
                (node, x - self.dx, 2 * y_join - y_done, -1)
                for node in self.dag.predecessors(node)
                if not self.dag.node[node]["done"]
            ]
        if dir == -1:
            todo_predecessors = [
                (node, x - self.dx, y_join + (s - 0.5) * self.dy, -1)
                for s, node in enumerate(self.dag.predecessors(node))
                if not self.dag.node[node]["done"]
            ]
        return todo_successors + todo_predecessors

    def _handle_fork(self, node, x, y, dir):
        assert self.dag.node[node]["fork"]
        y_fork = y if (dir == +1) else y + 0.5 * self.dy
        self.dag.node[node].update(x=x, y=y_fork, dir=dir, done=True)
        todo_predecessors = [
            (node, x - self.dx, y_fork, -1)
            for node in self.dag.predecessors(node)
            if not self.dag.node[node]["done"]
        ]
        if dir == -1:
            y_done = [
                self.dag.node[node]["y"]
                for node in self.dag.successors(node)
                if self.dag.node[node]["done"]
            ]
            assert len(y_done) == 1
            y_done = y_done[0]
            todo_successors = [
                (node, x + self.dx, 2 * y_fork - y_done, +1)
                for node in self.dag.successors(node)
                if not self.dag.node[node]["done"]
            ]
        if dir == +1:
            todo_successors = [
                (node, x + self.dx, y_fork + (s - 0.5) * self.dy, +1)
                for s, node in enumerate(self.dag.successors(node))
                if not self.dag.node[node]["done"]
            ]
        return todo_successors + todo_predecessors

    def _handle_line(self, node, x, y, dir):
        assert self.dag.node[node]["line"]
        while self.dag.node[node]["line"]:
            self.dag.node[node].update(x=x, y=y, dir=dir, done=True)
            x += dir * self.dx
            if (dir == +1) and self.dag.node[node]["leaf"]:
                return []
            if (dir == -1) and self.dag.node[node]["root"]:
                return []
            if (dir == +1):
                next_nodes = self.dag.successors(node)
            else:
                next_nodes = self.dag.predecessors(node)
            if len(next_nodes) != 1:
                raise ValueError(
                    f"x={x} y={y} dir={dir} node {node} next_nodes={next_nodes} data={self.dag.node[node]}")
            node = next_nodes[0]
        return [(node, x, y, dir)]

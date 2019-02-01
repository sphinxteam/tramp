from .dag_model import DAGModel
from .dag_algebra import FactorDAG


class FactorModel(DAGModel):
    def __init__(self, factor_dag):
        if not isinstance(factor_dag, FactorDAG):
            raise TypeError(f"factor_dag {factor_dag} is not a FactorDAG")
        for node in factor_dag._roots_ph:
            raise ValueError(f"root node {node} not a prior")
        self.factor_dag = factor_dag
        model_dag = factor_dag.to_model_dag()
        DAGModel.__init__(self, model_dag)

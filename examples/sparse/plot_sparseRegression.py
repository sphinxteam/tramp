"""
Sparse linear regression
========================

This example will illustrate how to use different simple modules of TRAMP 
"""
# %%
from tramp.variables import SISOVariable as V, SILeafVariable as O
from tramp.priors import GaussBernouilliPrior
from tramp.likelihoods import GaussianLikelihood
from tramp.channels import GaussianChannel, LinearChannel, AnalyticalLinearChannel
from tramp.ensembles import GaussianEnsemble, MarchenkoPasturEnsemble
from tramp.algos import CustomInit
from tramp.algos import ExpectationPropagation, StateEvolution, EarlyStopping
from tramp.algos.metrics import mean_squared_error

# %%
# Define a Sparse teacher


class SparseTeacher():
    def __init__(self, N, alpha, rho, Delta):
        self.N = N
        self.alpha = alpha
        self.M = int(self.alpha * self.N)
        self.rho = rho
        self.Delta = Delta
        self.model = self.build_model()

    def build_model(self):
        self.prior = GaussBernouilliPrior(size=(self.N,), rho=self.rho)
        ensemble = GaussianEnsemble(self.M, self.N)
        self.A = ensemble.generate()
        model = self.prior @ V(id="x") @ LinearChannel(W=self.A) @ V(
            id='z') @ GaussianChannel(var=self.Delta) @ O(id="y")
        model = model.to_model()
        return model


# %%
# Create a sparse teacher
N = 1000
alpha = 1
rho = 0.1
Delta = 1e-2

teacher = SparseTeacher(
    N=N, alpha=alpha, rho=rho, Delta=Delta)
sample = (teacher.model).sample()

# %%
prior = GaussBernouilliPrior(size=(N,), rho=rho)
student = prior @ V(id="x") @ LinearChannel(W=teacher.A) @ V(
    id='z') @ GaussianLikelihood(y=sample['y'], var=Delta)
student = student.to_model_dag()
student = student.to_model()

# %%
max_iter = 20
damping = 0.1

ep = ExpectationPropagation(student)
ep.iterate(
    max_iter=max_iter, damping=damping, callback=EarlyStopping(tol=1e-8))
data_ep = ep.get_variables_data(['x'])
mse = mean_squared_error(
    data_ep['x']['r'], sample['x'])
print(mse)


# %%

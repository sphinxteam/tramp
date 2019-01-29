from sklearn.metrics import mean_squared_error
from ..channels import GaussianChannel, AbsChannel, SngChannel
from ..likelihoods import GaussianLikelihood, AbsLikelihood, SngLikelihood
from ..algos import ExpectationPropagation, StateEvolution, EarlyStopping
from ..models import MultiLayerModel

def channel2likelihood(channel, y):
    if isinstance(channel, GaussianChannel):
        return GaussianLikelihood(var = channel.var, y=y)
    if isinstance(channel, AbsChannel):
        return AbsLikelihood(y=y)
    if isinstance(channel, SngChannel):
        return SngLikelihood(y=y)

# TODO implement for general DAG and signals to infer
class TeacherStudentScenario():
    """Implementation teacher student scenario.

    Parameters
    ----------
    model : MultiLayerModel instance
        generative model

    Notes
    -----
    In this impementation we assume that the signal x to infer is the first
    variable and the measurement y is the last variable.
    """

    def __init__(self, model):
        if not isinstance(model, MultiLayerModel):
            raise NotImplementedError(
                f"TeacherStudentScenario only implemented for MultiLayerModel"
            )
        self.teacher = model

    def setup(self):
        # teacher generate data
        sample = self.teacher.sample()
        self.true_values = sample
        self.x_true = sample[0]["X"]
        self.y = sample[-1]["X"]
        # pass it to the student
        last_channel = self.teacher.layers[-1]
        likelihood = channel2likelihood(last_channel, self.y)
        student_layers = self.teacher.layers[:-1] + [likelihood]
        self.student = MultiLayerModel(student_layers)

    def infer(self, callback=None, initializer=None):
        callback = callback or EarlyStopping(tol=1e-6, min_variance=1e-12)
        # run EP
        ep = ExpectationPropagation(self.student)
        ep.iterate(max_iter=250, callback=callback, initializer=initializer)
        self.x_pred = ep.get_variables_data()[0]["r"]
        self.mse_ep = mean_squared_error(self.x_true, self.x_pred)
        self.n_iter_ep = ep.n_iter
        # run SE
        se = StateEvolution(self.student)
        se.iterate(max_iter=250, callback=callback, initializer=initializer)
        self.mse_se = se.get_variables_data()[0]["v"]
        self.n_iter_se = se.n_iter

import numpy as np
from numpy import sqrt, abs
from numpy.linalg import inv, norm
from scipy.linalg import sqrtm
import copy


class SE_matrix_factorization(object):
    def __init__(self, K=1, N=1000, M=1000, model='UV', au_av=[1, 1], ax=1,  verbose=False):

        # Parameters
        self.model = model  # Model 'XX' or 'UV'
        self.K, self.N, self.M = K, N, M
        assert N != 0
        if self.model == 'XX':
            self.M = self.N
        self.alpha = M / N  # alpha = M/N

        # Options
        self.verbose = verbose

        # Variance and means priors
        bu, bv = np.zeros((self.K, self.M)), np.zeros((self.K, self.N))
        au, av = au_av
        self.Sigma_u, self.Sigma_v, self.lambda_u, self.lambda_v = 1/au * \
            np.identity(self.K), 1/av * \
            np.identity(self.K), (bu/au), (bv/av)
        if self.model == 'XX':
            self.Sigma_u = self.Sigma_v
            self.lambda_u = self.lambda_v
        self.Delta = 1/ax

        self.initialization_coef_d = 0.9
        self.initialization_coef_a = 0.1
        self.idty = np.identity(self.K)
        self.data = {"qu": [], "qv": []}
        self.qu, self.qv = 0, 0

        self.step = 0
        self.max_steps = 10000
        self.min_steps = 50
        self.precision = 1e-5
        self.diff = 0
        self.damping_coef = 0

    def initialization(self):
        self.qu = (self.initialization_coef_d - self.initialization_coef_a) * np.identity(self.K) + self.initialization_coef_a * \
            np.ones((self.K, self.K)) * np.random.randn()
        self.qv = (self.initialization_coef_d - self.initialization_coef_a) * np.identity(self.K) + self.initialization_coef_a * \
            np.ones((self.K, self.K)) * np.random.randn()
        self.add_data_to_dict()

    def SP_qv(self):
        if self.model == 'UV':
            gamma_u = self.alpha * self.qu / self.Delta
        elif self.model == 'XX':
            gamma_u = self.qv / self.Delta

        term_1 = inv(self.Sigma_v + gamma_u)
        term_2 = inv(self.Sigma_v) @ self.lambda_v @ self.lambda_v.T @ inv(
            self.Sigma_v)
        term_3 = gamma_u
        term_4 = gamma_u @ self.Sigma_v @ gamma_u.T
        term_5 = gamma_u @ self.lambda_v @ self.lambda_v.T @ gamma_u
        term_6 = 2 * \
            inv(self.Sigma_v) @ self.lambda_v @ self.lambda_v.T @ gamma_u

        res = term_1 @ (term_2 + term_3 + term_4 + term_5 + term_6) @ term_1
        return res

    def SP_qu(self):
        gamma_v = self.qv / self.Delta
        term_1 = inv(self.Sigma_u + gamma_v)
        term_2 = inv(self.Sigma_u) @ self.lambda_u @ self.lambda_u.T @ inv(
            self.Sigma_u)
        term_3 = gamma_v
        term_4 = gamma_v @ self.Sigma_u @ gamma_v.T
        term_5 = gamma_v @ self.lambda_u @ self.lambda_u.T @ gamma_v
        term_6 = 2 * \
            inv(self.Sigma_u) @ self.lambda_u @ self.lambda_u.T @ gamma_v

        res = term_1 @ (term_2 + term_3 + term_4 + term_5 + term_6) @ term_1
        return res

    def iteration(self):
        self.step += 1
        qv = self.SP_qv()
        if self.model == 'UV':
            qu = self.SP_qu()
        elif self.model == 'XX':
            qu = qv
        self.qv = self.damping(qv, copy.copy(self.qv))
        self.qu = self.damping(qu, copy.copy(self.qu))
        self.add_data_to_dict()
        self.print_last_iteration()

    def stopping_criteria(self):
        if self.step > 1:
            tab_name = ["qu", "qv"]
            tab_diff = np.array(
                [norm(self.data[str][-1]-self.data[str][-2])/norm(self.data[str][-1]) for str in tab_name])
            m = max(tab_diff)
            self.diff = m
        else:
            self.diff = 0

        if self.step < self.min_steps:
            return False
        else:
            if m < self.precision:
                return True
            else:
                return False

    def main(self):
        self.initialization()
        print('Delta = ', self.Delta) if self.verbose else 0
        while not self.stopping_criteria() and self.step < self.max_steps:
            self.iteration()
        self.qu = self.data["qu"][-1]
        self.qv = self.data["qv"][-1]
        self.compute_MSE()
        return np.sum(self.MSE_u), np.sum(self.MSE_v)

    def compute_MSE(self):
        MSE_v, MSE_u = self.Sigma_v - self.qv, self.Sigma_u - self.qu
        self.MSE_u = MSE_u
        self.MSE_v = MSE_v

    def damping(self, new_obj, old_obj):
        res = new_obj * (1-self.damping_coef) + old_obj * (self.damping_coef)
        return res

    def add_data_to_dict(self):
        self.data["qu"].append(self.qu)
        self.data["qv"].append(self.qv)

    def print_last_iteration(self):
        if self.verbose:
            print(
                f'Step: {self.step} Diff:{self.diff:.2e} qu: {self.data["qu"][-1]} qv:{self.data["qv"][-1]}')

import numpy as np
from numpy import sqrt, abs
from copy import deepcopy
from numpy.linalg import inv, norm

class VAMP_matrix_factorization(object):
    def __init__(self, K=1, N=1000, M=1000, model='UV', Delta=1, 
            au_av_bu_bv=[1, 1, np.zeros((1, 1000)),  np.zeros((1, 1000))], ax_bx=[1, np.ones((1000, 1000))],  
            seed=False, verbose=False, max_iter='inf', test_mode=False, initialization_mode='deterministic'):
        # Parameters
        self.model = model  # Model 'XX' or 'UV'
        self.K, self.N, self.M = K, N, M
        assert N != 0
        if self.model == 'XX':
            self.M = self.N
        self.alpha = M / N  # alpha = M/N
        self.Delta = Delta

        np.random.seed(0) if seed else 0  # Seed
        self.verbose = verbose

        ## Variance and means data
        self.lambda_X, self.Sigma_X = 0, 1

        ## Variance and means priors
        au, av, bu, bv = au_av_bu_bv
        self.Sigma_u, self.Sigma_v, self.lambda_u, self.lambda_v = 1/au * np.identity(self.K), 1/av * np.identity(self.K), (bu/au).T, (bv/av).T

        self.test_mode = test_mode
        if test_mode:
            ax, bx = self.generate_ax_bx()
        else:
            ax, bx = ax_bx
            self.U = np.zeros((self.K, self.M))
            self.V = np.zeros((self.K, self.N))
        self.Y, self.Delta = bx/ax, 1/ax
        
        ## Storage 
        self.list_evolution_m_q = []
        self.m_v, self.q_v, self.m_u, self.q_u = 0, 0, 0, 0
        self.tab_m_q = []

        # Convergence of the algorithm
        self.min_step_AMP = 25
        if max_iter == 'inf':
            self.max_step_AMP = 500
        else:
            self.max_step_AMP = max_iter
        self.diff_mode = 'overlap'
        self.threshold_error_overlap = 1e-5
        # Initialization
        self.initialization_mode = initialization_mode  # random, planted, deterministic
        self.planted_noise, self.coef_initialization_random = 1, 1
        # Damping
        self.damping_activated, self.damping_coef = True, 0.1

        if self.verbose:
            print(f'Model: {self.model}')
            print(f'K: {self.K} N: {self.N} M:{self.M}')
            print(f"alpha: {self.alpha}")
            print(f"Delta: {self.Delta}")
            print(f"Seed: {seed}")
            print(f'Initialization: {self.initialization_mode}')
            print(f'Damping: {self.damping_activated} / {self.damping_coef}')

    def generate_U_V(self):
        """
        Generate teacher weights U^*, V^*
        """
        self.U = np.zeros((self.K, self.M))
        self.V = np.zeros((self.K, self.N))
        for i in range(self.N):
            self.V[:, i] = np.random.multivariate_normal(
                self.lambda_v[:, i], self.Sigma_v)
        if self.model == 'UV':
            for j in range(self.M):
                self.U[:, j] = np.random.multivariate_normal(
                    self.lambda_u[:, j], self.Sigma_u)
        elif self.model == 'XX':
            self.U = self.V

    def Phi_out(self, z):
        """
        Returns Phi(z1) + N(0,Delta_1)
        """
        resul = z
        size = resul.shape
        noise = np.random.normal(0, sqrt(self.Delta), size)
        if self.model == 'XX':
            resul += (noise + noise.T) / sqrt(2)
        elif self.model == 'UV':
            resul += noise
        return resul

    def generate_Y(self):
        """
        Generate output Y
        - self.model == 'XX' : Y = VV/sqrt(N)'
        - self.model == 'UV' : Y = UV/sqrt(N)'
        """
        self.Y = np.zeros((self.M, self.N))
        self.generate_U_V()
        if self.model == 'UV':
            Z = (self.U.T.dot(self.V)) / sqrt(self.N)
        elif self.model == 'XX':
            Z = (self.V.T.dot(self.V)) / sqrt(self.N)
        Y = self.Phi_out(Z)
        return Y

    def generate_ax_bx(self):
        self.Y = self.generate_Y()
        ax, bx = 1/self.Delta, self.Y / self.Delta
        return ax, bx

    def generate_S_R(self):
        """
        Defines S and R if self.non_linearity == 'linear'
        Add a threshold to avoid overflow
        """
        Delta = max(self.Delta, 1e-2)
        self.S = (self.Y / Delta).T
        self.S_square = (np.square(self.S))
        self.R = (- 1 / Delta * np.ones((self.M, self.N)) + self.S_square.T).T

    ############## Initialization ##############
    def initialization(self):
        """
        Initialization of the messages
        """
        ###### Initilization ######
        # Initialization Vhat, Chat_V, Uhat, Chat_U at t=0
        self.initialization_Vhat_Chat_V()
        self.initialization_Uhat_Chat_U()
        self.compute_overlap()
        # Initilization A_U, V, A_V at t=0 and B_U, B_V, omega WITHOUT ONSAGER term
        self.initialization_no_Onsager_terms = True
        self.initialization_B_A_V()
        self.initialization_B_A_U()
        self.initialization_no_Onsager_terms = False
        # Store Onsager
        self.V_hat_onsager = deepcopy(self.V_hat)
        self.U_hat_onsager = deepcopy(self.U_hat)
        # Update
        (self.V_hat, self.C_hat_V) = self.update_Vhat_Chat_V()
        (self.U_hat, self.C_hat_U) = self.update_Uhat_Chat_U()
        self.compute_overlap()

        self.break_AMP, self.m_q_close_bayes = False, False
        self.step, self.diff = 0, 0
        self.threshold_error = self.threshold_error_overlap
        self.diff = self.threshold_error * 10
        self.list_diff = []

    # Initialization U_hat C_hat_U, V_hat C_hat_V
    def initialization_Uhat_Chat_U(self):
        """
        Initialization U_hat C_hat_U
        """
        if self.model == 'XX':
            self.U_hat = self.V_hat
            self.C_hat_U = self.C_hat_V
        elif self.model == 'UV':
            U_hat = np.zeros((self.K, self.M))
            C_hat_U = np.zeros((self.K, self.K, self.M))
            for mu in range(self.M):
                U_hat[:, mu] = self.initialization_Uhat(mu)
                C_hat_U[:, :, mu] = self.initialization_Chat_U()
            self.U_hat = U_hat
            self.C_hat_U = C_hat_U

    def initialization_Uhat(self, mu):
        """
        Initializes U_hat
        - deterministic : uses deterministic initialization
        - random : random vector
        - planted : ground truth + noise
        """
        if self.initialization_mode == 'deterministic':
            U_hat = 0.1 * np.ones(self.K)
        elif self.initialization_mode == 'random':
            U_hat = self.coef_initialization_random * np.random.random(self.K)
        elif self.initialization_mode == 'planted':
            U_hat = self.U[:, mu] + self.planted_noise * \
                np.random.random(self.K)
        else:
            raise NameError('Wrong intialization U_hat')
        return U_hat

    def initialization_Chat_U(self):
        """
        Initializes C_hat_U with variance of the prior on U
        """
        C_hat_U = 0.01 * np.identity(self.K)
        return C_hat_U

    def initialization_B_A_U(self):
        """
        Initializes B_U, A_U
        updating B_U, A_U without the Onsager term
        """
        self.B_U = self.update_B_U()
        self.A_U = np.abs(self.update_A_U())

    def initialization_Vhat_Chat_V(self):
        """
        Initialization  V_hat C_hat_V
        """
        V_hat = np.zeros((self.K, self.N))
        C_hat_V = np.zeros((self.K, self.K, self.N))
        for j in range(self.N):
            V_hat[:, j] = self.initialization_Vhat(j)
            C_hat_V[:, :, j] = self.initialization_Chat_V()
        self.V_hat = V_hat
        self.C_hat_V = C_hat_V

    def initialization_Vhat(self, j):
        """
        Initializes V_hat
        - deterministic : uses deterministic initialization
        - random : random vector
        - planted : ground truth + noise
        """
        if self.initialization_mode == 'deterministic':
            V_hat = 0.1 * np.ones(self.K)
        elif self.initialization_mode == 'random':
            V_hat = self.coef_initialization_random * np.random.random(self.K)
        elif self.initialization_mode == 'planted':
            V_hat = self.V[:, j] + self.planted_noise * \
                np.random.random(self.K)
        else:
            raise NameError('Wrong intialization V_hat')
        return V_hat

    def initialization_Chat_V(self):
        """
        Initializes C_hat_V with variance of the prior on V
        """
        C_hat_V = 0.01 * np.identity(self.K)
        return C_hat_V

    def initialization_B_A_V(self):
        """
        Initializes B_V, A_V
        updating B_V, A_V without the Onsager term
        """
        self.B_V = self.update_B_V()
        self.A_V = self.update_A_V()

    ############## Updates ##############
    def update_Uhat_Chat_U(self):
        """
        Updates U_hat and Chat_U
        """
        U_hat = np.zeros((self.K, self.M))
        C_hat_U = np.zeros((self.K, self.K, self.M))
        for mu in range(self.M):
            U_hat[:, mu], C_hat_U[:, :, mu] = self.fU_fC(mu)
        return (U_hat, C_hat_U)

    def fU_fC(self, mu):
        """
        Computes fU and fC:
            fU = int du int dz u P_out(u|z) N_u( , ) N_z( , )
            fC = int du int dz u^2 P_out(u|z) N_u( , ) N_z( , ) - fU^2
        """
        A_U, B_U = self.A_U[:, :, mu], self.B_U[:, mu]
        sigma_star = inv(inv(self.Sigma_u)+A_U)
        lambda_star = sigma_star.dot(
            inv(self.Sigma_u).dot(self.lambda_u[:, mu]) + B_U)
        fU = lambda_star
        fC = sigma_star
        return fU, fC

    def update_B_U(self):
        """
        Updates B_U
        """
        B_U = 1/sqrt(self.N) * np.einsum('jm,kj->km', self.S, self.V_hat)
        if not self.initialization_no_Onsager_terms:
            B_U -= 1/self.N * np.einsum(
                'klm,lm->km', np.einsum('jm,klj->klm', self.S_square, self.C_hat_V), self.U_hat_onsager)
        return B_U

    def update_A_U(self):
        """
        Updates A_U
        """
        A_U = 1/self.N * (np.einsum('jm,klj->klm', self.S_square - self.R, np.einsum(
            'kj,lj->klj', self.V_hat, self.V_hat)) - np.einsum('jm,klj->klm', self.R, self.C_hat_V))
        return A_U

    def update_Vhat_Chat_V(self):
        """
        Updates U_hat and Chat_U
        If self.model == 'UU' : V_hat = U_hat
        """
        if self.model == 'UU':
            V_hat = self.U_hat
            C_hat_V = self.C_hat_U
        else:
            V_hat = np.zeros((self.K, self.N))
            C_hat_V = np.zeros((self.K, self.K, self.N))
            for j in range(self.N):
                V_hat[:, j], C_hat_V[:, :, j] = self.fV_fC(j)
        return (V_hat, C_hat_V)

    def fV_fC(self, j):
        """
        Computes fV and fC:
            fV = int dv v P_v(v) N_v( , ) 
            fC = int dv v^2 P_v(v) N_v( , ) - fV^2
        """
        A_V, B_V = self.A_V[:, :, j], self.B_V[:, j]
        sigma_star = inv(inv(self.Sigma_v)+A_V)
        lambda_star = sigma_star.dot(
            inv(self.Sigma_v).dot(self.lambda_v[:, j]) + B_V)
        fV = lambda_star
        fC = sigma_star
        return (fV, fC)

    def update_B_V(self):
        """
        Updates B_V
        """
        B_V = 1/sqrt(self.N) * np.einsum('jm,km->kj', self.S, self.U_hat)
        if not self.initialization_no_Onsager_terms:
            B_V -= 1/self.N * np.einsum(
                'klj,lj->kj', np.einsum('jm,klm->klj', self.S_square, self.C_hat_U), self.V_hat_onsager)
        return B_V

    def update_A_V(self):
        """
        Updates A_V
        """
        A_V = 1/self.N * (np.einsum('jm,klm->klj', self.S_square-self.R, np.einsum(
            'km,lm->klm', self.U_hat, self.U_hat)) - np.einsum('jm,klm->klj', self.R, self.C_hat_U))
        return A_V

    ########### Overlap, MSE, diff ###########
    def compute_overlap(self):
        """ 
        Compute overlap parameters
        q_X = 1/size * X' X
        m_X = 1/size * X' X0
        """
        m_v = 1/self.N * np.abs(self.V_hat.dot(self.V.T))
        q_v = 1/self.N * self.V_hat.dot(self.V_hat.T)
        m_u = 1/self.M * np.abs(self.U_hat.dot(self.U.T))
        q_u = 1/self.M * self.U_hat.dot(self.U_hat.T)

        # Store old overlaps
        self.m_v_old, self.q_v_old, self.m_u_old, self.q_u_old = self.m_v, self.q_v, self.m_u, self.q_u
        # Update them
        self.m_v, self.q_v, self.m_u, self.q_u = m_v, q_v, m_u, q_u
        # Print
        if self.verbose:
            if self.K == 1:
                print(
                    f'm_u: {self.m_u.item():.3f} m_v: {self.m_v.item():.3f}')
                print(
                    f'q_u: {self.q_u.item():.3f} q_v: {self.q_v.item():.3f}')
            else:
                print(
                    f'm_u: {self.m_u} m_v:{self.m_v}')
                print(
                    f'q_u: {self.q_u} q_v:{self.q_v}')
        list_m_q = [self.m_v, self.m_u, self.q_v, self.q_u]
        self.list_evolution_m_q.append(list_m_q)
        self.tab_m_q = np.array(list_m_q)
        return self.tab_m_q

    def compute_MSE(self):
        """ 
        Compute the MSE in Bayes Optimal setting
        MSE = Q^0_X - q_X = 1 - 2 q_X + m_X 
        """
        MSE_v = 1 - 2 * self.m_v + self.q_v
        MSE_u = 1 - 2 * self.m_u + self.q_u

        self.MSE_v, self.MSE_u = MSE_v, MSE_u
        self.tab_MSE = [MSE_v, MSE_u]
        
        if self.verbose:
            if self.K == 1:
                print(f'MSE_u = {self.MSE_u.item() :.3f} MSE_v = {self.MSE_v.item() :.3f}')
            else :
                print(
                    f'MSE_u = {self.MSE_u} MSE_v = {self.MSE_v}')

    def compute_difference(self):
        """ 
        Compute difference between t and t+1 of :
        - overlaps | q_X^{t} - q_X^{t+1} |
        """
        if self.K == 1:
            diff_overlaps = max([np.abs(
                self.q_v-self.q_v_old), np.abs(self.q_u-self.q_u_old)])
        else:
            diff_overlaps = max([np.abs(norm(self.q_v-self.q_v_old)), np.abs(
                norm(self.q_u-self.q_u_old))]) / (self.K**2)
        if self.step > self.min_step_AMP:
            self.diff = diff_overlaps
        self.list_diff.append(diff_overlaps)
        if self.verbose:
            print(f'Diff {self.diff_mode}: {diff_overlaps.item():.4e}')
    
    # Damping
    def damping(self, X_new, X_self):
        """
        if damping activated returns X_new
        else returns (1-self.damping_coef) * (X_new) + self.damping_coef * X_self
        """
        if not self.damping_activated:
            return X_new
        else:
            return (1-self.damping_coef) * (X_new) + self.damping_coef * X_self

    ############## AMP training ##############
    def check_break_AMP(self):
        """
        Reasons to break AMP iterations
        - cond1: takes too long
        - cond2: preicsion low enough
        """
        # If takes too long
        cond_1 = self.step > self.max_step_AMP
        # If precision high enough and reaches Bayes optimality q=m
        cond_2 = self.diff < self.threshold_error
        list_cond = [cond_1, cond_2]
        if any(list_cond):
            self.break_AMP = True
            print(f'Breaking conditions: {list_cond}') if self.verbose else 0

    def AMP_step(self):
        """ 
        One step of AMP iteration
        """
        self.step += 1
        # Layer Matrix factorization
        # A_V(t) <- requires: U_hat(t), C_hat_U(t)
        A_V = self.update_A_V()
        self.A_V = self.damping(A_V, self.A_V)
        # B_V(t) <- requires: U_hat(t), C_hat_U(t), V_hat(t-1)
        B_V = self.update_B_V()
        self.B_V = self.damping(B_V, self.B_V)
        # A_U(t) <- requires: V_hat(t), C_hat_V(t)
        A_U = self.update_A_U()
        self.A_U_onsager = deepcopy(self.A_U)
        self.A_U = self.damping(A_U, self.A_U)
        # B_U(t) <- requires: V_hat(t), C_hat_V(t), U_hat(t-1)
        B_U = self.update_B_U()
        self.B_U_onsager = deepcopy(self.B_U)
        self.B_U = self.damping(B_U, self.B_U)

        # Onsager
        self.U_hat_onsager = deepcopy(self.U_hat)
        self.V_hat_onsager = deepcopy(self.V_hat)

        # Update V_hat(t+1) <- requires: B_V(t), A_V(t)
        (self.V_hat, self.C_hat_V) = self.update_Vhat_Chat_V()
        # Update U_hat(t+1) <- requires: B_U(t), A_U(t), omega(t), V(t)
        (self.U_hat, self.C_hat_U) = self.update_Uhat_Chat_U()

    def VAMP_training(self):
        """
        Iterate AMP with priors on u, v coming from VAMP messages  
        """
        self.generate_S_R()
        self.initialization()

        while not self.break_AMP:
            print(f'Step = {self.step}') if self.verbose else 0
            self.AMP_step()
            self.compute_overlap()
            if self.test_mode:
                self.compute_MSE()
            self.compute_difference()
            self.check_break_AMP()
        
        ## Transform tensor C_hat_U, C_hat_V in scalar
        ## v_u = 1/(MK) sum_{mu k} E[u_{mu k}^2] - E[u_{mu k}]^2 
        rz_u, rz_v = self.U_hat.T, self.V_hat.T
        vz_u = 1/(self.M * self.K) * np.sum([ np.trace(self.C_hat_U[:,:,mu]) for mu in range(self.M) ])
        vz_v = 1/(self.N * self.K) * np.sum([ np.trace(self.C_hat_V[:,:,i]) for i in range(self.N) ])

        assert rz_u.shape == (self.M, self.K)
        assert rz_v.shape == (self.N, self.K)
        assert vz_u.shape == ()
        assert vz_v.shape == ()

        return rz_u, vz_u, rz_v, vz_v

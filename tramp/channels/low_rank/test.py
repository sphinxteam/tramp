from .SE_matrix_factorization import SE_matrix_factorization
import numpy as np

N, M = 1000, 1000
Sigma_x = 0.5
Delta = 3.99

K = 3
au_av = [1/Sigma_x, 1/Sigma_x]
ax = 1 / Delta

SE = SE_matrix_factorization(
    K=K, N=N, M=M, model='XX', au_av=au_av, ax=ax, verbose=False)

mse_u, mse_v = SE.main()
print(mse_u, mse_v)

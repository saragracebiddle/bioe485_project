import numpy as np

def compute_tau_avg(coefs, dt, basis_array, N):
    k = np.arange(N)
    
    h = basis_array @ coefs
    b = sum(k*h)
    tau = dt * b / sum(h)
    return tau
import numpy as np

def compute_lqg_analytic_cost(P, y, Sigma, N, T):
    """
    Computes the exact theoretical cost for the LQG problem.
    Formula: y^T P(0) y + integral(Trace(Sigma^T P(t) Sigma))
    """
    dt = T / N
    
    # 1. Deterministic base cost
    deterministic_cost = float(y.T @ P[0] @ y)
    
    # 2. Accumulate the noise penalty over time
    noise_penalty = 0.0
    for i in range(N):
        noise_penalty += np.trace(Sigma.T @ P[i] @ Sigma) * dt
        
    return deterministic_cost + noise_penalty
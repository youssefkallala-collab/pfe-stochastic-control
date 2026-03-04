import numpy as np

def _ensure_symmetric(mat):
    """Helper to maintain numerical stability by forcing matrix symmetry."""
    return 0.5 * (mat + mat.T)

def dre_derivative(P, A, B, R_inv, Q):
    """
    Calculates the derivative dP/dt for the Differential Riccati Equation.
    Equation: A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q
    """
    return A.T @ P + P @ A - P @ B @ R_inv @ B.T @ P + Q

def solve_dre(A, B, Q, R, S, N, T, method='rk4'):
    """
    Unified solver for the Differential Riccati Equation (solved backward in time).
    
    Parameters:
        A, B, Q, R: System matrices
        S: Terminal cost matrix P(T)
        N: Number of time steps
        T: Total time horizon
        method: 'euler', 'rk2', or 'rk4'
    """
    dt = T / N
    steps = np.zeros((N + 1, A.shape[0], A.shape[1]))
    steps[N] = S
    R_inv = np.linalg.inv(R)
    
    # Common arguments for the derivative function
    args = (A, B, R_inv, Q)

    for i in range(N, 0, -1):
        P_curr = steps[i]
        
        if method == 'euler':
            k1 = dre_derivative(P_curr, *args)
            P_next = P_curr + k1 * dt
            
        elif method == 'rk2':
            k1 = dre_derivative(P_curr, *args)
            P_mid = _ensure_symmetric(P_curr + k1 * (dt / 2.0))
            k2 = dre_derivative(P_mid, *args)
            P_next = P_curr + k2 * dt
            
        elif method == 'rk4':
            k1 = dre_derivative(P_curr, *args)
            k2 = dre_derivative(_ensure_symmetric(P_curr + k1 * (dt / 2.0)), *args)
            k3 = dre_derivative(_ensure_symmetric(P_curr + k2 * (dt / 2.0)), *args)
            k4 = dre_derivative(_ensure_symmetric(P_curr + k3 * dt), *args)
            P_next = P_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        else:
            raise ValueError(f"Unknown method: {method}")

        steps[i-1] = _ensure_symmetric(P_next)
        
    return steps
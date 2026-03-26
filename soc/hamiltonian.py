import numpy as np

def get_u_star(B, R, lam, gamma=1.0):
    """
    Computes the optimal control minimizer for the LQ Hamiltonian.
    Formula: u* = -(1 / 2*gamma) * R^{-1} B^T \lambda
    """
    R_inv = np.linalg.inv(R)
    return (-0.5 / gamma) * R_inv @ B.T @ lam

def compute_H(x, lam, V, gamma, A, B, Q, R, Sigma):
    """
    Computes the scalar value of the Extended Hamiltonian for the LQ problem.
    """
    u_star = get_u_star(B, R, lam, gamma)
    
    # 1. Costate term: \lambda^T (Ax + Bu)
    drift_term = lam.T @ (A @ x + B @ u_star)
    
    # 2. Hessian term: 0.5 * Trace(\sigma \sigma^T V)
    diffusion_term = 0.5 * np.trace(Sigma @ Sigma.T @ V)
    
    # 3. Running cost term: \gamma * (x^T Q x + u^T R u)
    cost_term = gamma * (x.T @ Q @ x + u_star.T @ R @ u_star)
    
    return drift_term + diffusion_term + cost_term

# ==========================================
# HAMILTONIAN PARTIAL DERIVATIVES (DANSKIN'S THEOREM)
# ==========================================

def dH_dlam(x, u_star, A, B, gamma=1.0):
    """
    Derivative of H with respect to the costate \lambda.
    Yields the induced drift vector.
    """
    return A @ x + B @ u_star

def dH_dV(Sigma, gamma=1.0):
    """
    Derivative of H with respect to the Hessian V.
    Yields half of the diffusion metric matrix.
    """
    return 0.5 * (Sigma @ Sigma.T)

def dH_dgamma(x, u_star, Q, R, gamma=1.0):
    """
    Derivative of H with respect to the cost weight \gamma.
    Yields the induced running cost scalar.
    """
    return x.T @ Q @ x + u_star.T @ R @ u_star
import numpy as np
import torch

def get_u_star(B, R, lam, gamma=1.0):
    """
    Computes the optimal control minimizer for the LQ Hamiltonian.
    Formula: u* = -(1 / 2*gamma) * R^{-1} B^T lambda
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
    Derivative of H with respect to the costate lambda.
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
    Derivative of H with respect to the cost weight gamma.
    Yields the induced running cost scalar.
    """
    return x.T @ Q @ x + u_star.T @ R @ u_star

def compute_hamiltonian_partials(cfg, x, V_x, Hessian):
    """
    Used by simulate.py (Evaluation Phase).
    Returns the derivatives of the Hamiltonian to drive the stochastic simulation.
    """
    A = cfg.A
    B = cfg.B
    Q = cfg.Q
    R = cfg.R
    Sigma = cfg.Sigma
    R_inv = torch.inverse(R)
    
    # Calculate optimal control
    u_star = -0.5 * R_inv @ B.T @ V_x
    
    # 1. dH/dlambda (The Optimal Drift: Ax + Bu*)
    batch_size = x.shape[0]
    A_batched = A.expand(batch_size, -1, -1)
    B_batched = B.expand(batch_size, -1, -1)
    d_lambda_H = torch.bmm(A_batched, x) + torch.bmm(B_batched, u_star)
    
    # 2. dH/dV (The Diffusion Covariance: 0.5 * Sigma * Sigma^T)
    # Note: V here refers to the Hessian matrix in the PDF's math notation.
    Sigma_batched = Sigma.expand(batch_size, -1, -1)
    d_V_H = 0.5 * torch.bmm(Sigma_batched, Sigma_batched.transpose(1, 2))
    
    # 3. dH/dgamma (The Running Cost: x^TQx + u^TRu)
    # Even though we don't accumulate this in the PINN simulator, we return it for mathematical completeness
    Q_batched = Q.expand(batch_size, -1, -1)
    R_batched = R.expand(batch_size, -1, -1)
    state_cost = torch.bmm(torch.bmm(x.transpose(1, 2), Q_batched), x)
    control_cost = torch.bmm(torch.bmm(u_star.transpose(1, 2), R_batched), u_star)
    d_gamma_H = state_cost + control_cost
    
    return d_lambda_H, d_V_H, d_gamma_H


def compute_hjb_residual(cfg, x, v_t, v_x, hessian):
    """
    Assembles the optimal Hamiltonian H* and computes the HJB residual.
    Residual = V_t + H* = V_t + tr(d_V_H * Hessian) + ∇V^T * d_lambda_H + d_gamma_H
    """
    # 1. Get the physics/system dynamics partials
    d_lambda_H, d_V_H, d_gamma_H = compute_hamiltonian_partials(cfg, x, v_x, hessian)

    # 2. Assemble H*
    # Diffusion: tr(d_V_H * Hessian)
    diffusion_term = torch.einsum('bij,bij->b', d_V_H, hessian).unsqueeze(-1)                      
    
    # Drift: ∇V^T * d_lambda_H
    drift_term = torch.bmm(v_x.transpose(1, 2), d_lambda_H).squeeze(-1) 
    
    # State & Control Cost
    state_control_cost = d_gamma_H.squeeze(-1) 

    H_star = diffusion_term + drift_term + state_control_cost

    # 3. Compute the final residual (which we want to drive to 0)
    hjb_residual = v_t + H_star 
    
    return hjb_residual

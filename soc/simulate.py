import numpy as np
import torch
import math
# Ensure you are importing your partials functions
from soc.potential import compute_partials
from soc.hamiltonian import compute_hamiltonian_partials

def lqr_system_dynamics(x, P_curr, A, B, Q, R, R_inv):
    """
    Calculates the derivatives for both state (dx/dt) and cost (dJ/dt).
    """
    # Compute optimal control: u = -R^-1 * B^T * P * x
    u = -R_inv @ B.T @ P_curr @ x
    
    # State derivative: dx/dt = Ax + Bu
    dx = A @ x + B @ u
    
    # Cost derivative: dJ/dt = x^T Q x + u^T R u
    dj = x.T @ Q @ x + u.T @ R @ u
    
    return dx, np.squeeze(dj)

def simulate_lqr(A, B, Q, R, S, P, y, N, T, method='rk4'):
    """
    Unified simulator for LQR state and cost accumulation.
    
    Parameters:
        P: Array of Riccati matrices. 
           Note: For 'rk4', P should have (2N + 1) entries to provide midpoints.
    """
    dt = T / N
    X = np.zeros((N + 1, len(y)))
    J = np.zeros(N + 1)
    X[0] = y
    
    R_inv = np.linalg.inv(R)
    args = (A, B, Q, R, R_inv)

    for i in range(N):
        x_curr = X[i]
        
        if method == 'euler':
            dx, dj = lqr_system_dynamics(x_curr, P[i], *args)
            X[i+1] = x_curr + dx * dt
            J[i+1] = J[i] + dj * dt

        elif method == 'rk2':
            # Step 1: Start
            k1_x, k1_j = lqr_system_dynamics(x_curr, P[i], *args)
            
            # Step 2: Midpoint (Linear interpolation for P)
            P_mid = 0.5 * (P[i] + P[i+1])
            x_mid = x_curr + k1_x * (dt / 2.0)
            k2_x, k2_j = lqr_system_dynamics(x_mid, P_mid, *args)
            
            X[i+1] = x_curr + k2_x * dt
            J[i+1] = J[i] + k2_j * dt

        elif method == 'rk4':
            # P indices for RK4 assuming P is length 2N + 1
            idx = 2 * i
            k1_x, k1_j = lqr_system_dynamics(x_curr, P[idx], *args)
            
            x_m1 = x_curr + k1_x * (dt / 2.0)
            k2_x, k2_j = lqr_system_dynamics(x_m1, P[idx+1], *args)
            
            x_m2 = x_curr + k2_x * (dt / 2.0)
            k3_x, k3_j = lqr_system_dynamics(x_m2, P[idx+1], *args)
            
            x_e = x_curr + k3_x * dt
            k4_x, k4_j = lqr_system_dynamics(x_e, P[idx+2], *args)
            
            X[i+1] = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            J[i+1] = J[i] + (dt / 6.0) * (k1_j + 2*k2_j + 2*k3_j + k4_j)
        
        else:
            raise ValueError(f"Method {method} not recognized.")

    # Apply terminal cost P(T) = S at the final state
    J[-1] += X[N].T @ S @ X[N]
    
    return X, J
def simulate_lqg_euler_maruyama(A, B, R, P, y, Sigma, N, T, M):
    """Simulates M stochastic trajectories simultaneously using Euler-Maruyama."""
    dt = T / N
    d = len(y)
    r = Sigma.shape[1]
    
    X = np.zeros((N + 1, M, d))
    X[0, :, :] = y
    
    R_inv = np.linalg.inv(R)
    
    for i in range(N):
        # Optimal feedback matrix K at time i
        K = -R_inv @ B.T @ P[i]
        
        # Calculate control for all M trajectories
        u = X[i] @ K.T
        
        # Drift and Diffusion
        drift = X[i] @ A.T + u @ B.T
        Z = np.random.randn(M, r)
        diffusion = Z @ Sigma.T
        
        # Euler-Maruyama Step
        X[i+1] = X[i] + drift * dt + diffusion * np.sqrt(dt)
        
    return X

def compute_lqg_mc_costs(X, P, Q, R, S, B, N, T):
    """Calculates the accumulated cost for all M trajectories."""
    dt = T / N
    M = X.shape[1]
    R_inv = np.linalg.inv(R)
    
    costs = np.zeros(M)
    for i in range(N):
        K = -R_inv @ B.T @ P[i]
        u = X[i] @ K.T
        
        # Fast batched quadratic form
        state_cost = np.sum((X[i] @ Q) * X[i], axis=1)
        control_cost = np.sum((u @ R) * u, axis=1)
        costs += (state_cost + control_cost) * dt
        
    terminal_cost = np.sum((X[N] @ S) * X[N], axis=1)
    costs += terminal_cost
    
    return costs

def simulate_nn_policy_euler_maruyama(u_theta, cfg):
    """
    Simulates stochastic trajectories using the PINN Hamiltonian-Partials Reformulation.
    Drift = dH/dlambda, Diffusion = sqrt(2 dH/dV).
    No running cost is accumulated since the PINN solves the value directly.
    """
    # Initialize State Tensor: shape (Steps, Trajectories, State Dim, 1)
    X = torch.zeros(cfg.N + 1, cfg.M, cfg.state_dim, 1, device=cfg.device)
    
    # Broadcast the initial state to all M trajectories
    # Note: Using cfg.y or cfg.y_init based on your models.py naming
    X[0, :, :, 0] = cfg.y.unsqueeze(0).expand(cfg.M, -1)
    
    # Switch network to evaluation mode (disables dropout/batchnorm/gradient graphs)
    u_theta.eval() 
    
    for i in range(cfg.N):
        t_val = i * cfg.dt
        t_tensor = torch.full((cfg.M, 1), t_val, device=cfg.device)
        
        # We MUST set requires_grad=True to calculate spatial derivatives
        x_tensor = X[i].clone().detach().requires_grad_(True)
        
        # 1. Autodiff Partials (From soc.potential)
        # Extracts Value, Spatial Gradient (lambda), and Spatial Hessian
        _, _, grad_u, hessian_u = compute_partials(u_theta, t_tensor, x_tensor)        
        # Ensure Hessian is perfectly symmetric for numerical stability
        hessian_u = 0.5 * (hessian_u + hessian_u.transpose(1, 2))
        
        # 2. Hamiltonian Partials (From soc.hamiltonian)
        # We ignore the 3rd return value (d_gamma_H / Cost) completely!
        d_lambda_H, d_V_H, _ = compute_hamiltonian_partials(cfg, x_tensor, grad_u, hessian_u)
        
        # 3. Construct Diffusion Matrix from d_V_H
        # Diffusion G = 2.0 * dH/dV
        G = 2.0 * d_V_H
        G = 0.5 * (G + G.transpose(1, 2)) # Enforce symmetry for Cholesky
        
        # Calculate Sigma via Cholesky decomposition
        Sigma = torch.linalg.cholesky(G)
        
        # 4. Euler-Maruyama Noise
        xi = torch.randn(cfg.M, cfg.state_dim, 1, device=cfg.device)
        dW = math.sqrt(cfg.dt) * xi
        noise = torch.bmm(Sigma, dW) 
        
        # 5. Update state (Using d_lambda_H as the optimal Drift)
        X[i+1] = x_tensor.detach() + (d_lambda_H.detach() * cfg.dt) + noise
        
    # Re-enable training mode
    u_theta.train()
    
    # Return ONLY the trajectories (no cost tensor anymore)
    return X.detach()
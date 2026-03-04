import numpy as np

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
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'soc' module can be imported if running from the scripts/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from soc.lqr import solve_dre
from soc.models import BenchmarkConfig
from soc.potential import ValueNetwork
from soc.optimize import train_potential_network
from soc.simulate import simulate_nn_policy_euler_maruyama

def main():
    print("==================================================")
    print("  EXP 2: PINN HJB STOCHASTIC OPTIMAL CONTROL")
    print("==================================================")

    # 1. Setup & Reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    cfg = BenchmarkConfig()
    print(f"Device: {cfg.device} | Batch: {cfg.batch_size} | Epochs: {cfg.epochs}")
    
    # 2. Initialize the Physics-Informed Neural Network
    u_theta = ValueNetwork(cfg).to(cfg.device)
    
    # 3. Train the PINN (Solving the HJB PDE)
    print("\n--- Training Phase ---")
    loss_history = train_potential_network(u_theta, cfg)
    
    # 4. Simulate the Physical System using the learned PDE
    print("\n--- Simulation Phase ---")
    X_nn = simulate_nn_policy_euler_maruyama(u_theta, cfg)
    
    # ==================================================
    # 5. THESIS-READY PLOTTING (Theory vs NN Comparison)
    # ==================================================
    os.makedirs("figures", exist_ok=True)
    
    print("\n--- Evaluating Learned Models vs Exact Theory ---")
    u_theta.eval()
    
    # ---------------------------------------------------------
    # A. EXACT THEORETICAL SOLUTION
    # ---------------------------------------------------------
    A_np = cfg.A.cpu().numpy()
    B_np = cfg.B.cpu().numpy()
    Q_np = cfg.Q.cpu().numpy()
    R_np = cfg.R.cpu().numpy()
    S_np = cfg.S.cpu().numpy()
    Sigma_np = cfg.Sigma.cpu().numpy()
    
    # Solve Riccati backwards in time
    P_exact = solve_dre(A_np, B_np, Q_np, R_np, S_np, cfg.N, cfg.T, method='rk4')

    # Pre-compute the stochastic noise penalty integral
    noise_penalty = np.zeros(cfg.N)
    accumulated_noise = 0.0
    for i in reversed(range(cfg.N)):
        accumulated_noise += np.trace(Sigma_np.T @ P_exact[i] @ Sigma_np) * cfg.dt
        noise_penalty[i] = accumulated_noise

    # ---------------------------------------------------------
    # B. EVALUATE ALONG THE TRAJECTORY X_t
    # ---------------------------------------------------------
    t_grid = np.linspace(0, cfg.T, cfg.N)

    V_nn_trajectory = np.zeros(cfg.N)
    V_true_trajectory = np.zeros(cfg.N)
    grad_nn_trajectory = np.zeros(cfg.N)
    grad_true_trajectory = np.zeros(cfg.N)
    
    state_dim = cfg.state_dim 
    
    for i in range(cfg.N):
        # 1. Cleanly extract states and force shape to (batch_size, state_dim)
        # We detach and require_grad to compute autograd over the simulation states
        X_tensor = X_nn[i].view(-1, state_dim).detach()
        X_tensor.requires_grad_(True)
        X_t = X_tensor.detach().cpu().numpy() 
        actual_batch = X_t.shape[0] # Handle any batch size dynamically
        
        # 2. Exact Cost and Gradient along X_t
        # Cost Batch
        cost_batch = np.sum((X_t @ P_exact[i]) * X_t, axis=1) + noise_penalty[i]
        V_true_trajectory[i] = cost_batch.mean()
        
        # Gradient Batch
        grad_true_batch = 2.0 * (X_t @ P_exact[i])
        grad_true_norm_batch = np.linalg.norm(grad_true_batch, axis=1)
        grad_true_trajectory[i] = grad_true_norm_batch.mean()
        
        # 3. NN Cost and Gradient along X_t
        t_tensor = torch.full((actual_batch, 1), t_grid[i], dtype=torch.float32, device=cfg.device)
        V_nn_batch = u_theta(t_tensor, X_tensor)
        V_nn_trajectory[i] = V_nn_batch.mean().item()
        
        # Autograd for the batch
        grad_nn_batch = torch.autograd.grad(
            outputs=V_nn_batch, 
            inputs=X_tensor,
            grad_outputs=torch.ones_like(V_nn_batch),
            create_graph=False
        )[0]
        
        # Gradient norm across the batch
        grad_nn_norm_batch = torch.norm(grad_nn_batch, dim=1)
        grad_nn_trajectory[i] = grad_nn_norm_batch.mean().item()

    # ---------------------------------------------------------
    # C. GENERATE COMPARISON PLOTS
    # ---------------------------------------------------------
    
    # PLOT 1: TRAJECTORY POTENTIAL COMPARISON
    plt.figure(1, figsize=(8, 6))
    plt.plot(t_grid, V_true_trajectory, color='gray', linewidth=5, alpha=0.6, 
             label=f'Exact Theoretical Value \n[Initial: {V_true_trajectory[0]:.4f}]')
    plt.plot(t_grid, V_nn_trajectory, color='purple', linewidth=2.5, linestyle='--', 
             label=f'PINN Value \n[Initial: {V_nn_trajectory[0]:.4f}]')
    plt.title("Trajectory Potential: PINN vs. Exact Theory", fontsize=15, fontweight='bold')
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Potential $V(t, X_t)$", fontsize=12)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path_trajectory = "figures/exp2_trajectory_comparison.png"
    plt.savefig(save_path_trajectory, dpi=300, bbox_inches='tight')
    print(f"Success! Trajectory comparison plot saved to: {save_path_trajectory}")
    
    # PLOT 2: TRAJECTORY GRADIENT (POLICY) COMPARISON
    plt.figure(2, figsize=(8, 6))
    plt.plot(t_grid, grad_true_trajectory, color='lightcoral', linewidth=5, alpha=0.6, 
             label='Exact Theoretical Gradient')
    plt.plot(t_grid, grad_nn_trajectory, color='#1f77b4', linewidth=2.5, linestyle='--', 
             label='PINN Gradient')
    plt.title("Trajectory Gradient (Policy): PINN vs. Exact Theory", fontsize=15, fontweight='bold')
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel(r"Gradient Norm $|| \nabla_x V ||$", fontsize=12)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path_grad_trajectory = "figures/exp2_trajectory_gradient_comparison.png"
    plt.savefig(save_path_grad_trajectory, dpi=300, bbox_inches='tight')
    print(f"Success! Trajectory gradient comparison plot saved to: {save_path_grad_trajectory}")
    
    # Show both plots at the end!
    plt.show()

if __name__ == "__main__":
    main()
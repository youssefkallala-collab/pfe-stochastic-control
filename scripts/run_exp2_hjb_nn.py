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
    # B. EVALUATE STATIC POINT OVER TIME (Plots 1 & 2)
    # ---------------------------------------------------------
    t_grid = np.linspace(0, cfg.T, cfg.N)
    
    V_nn_vals = np.zeros(cfg.N)
    V_true_vals = np.zeros(cfg.N)
    grad_nn_norm = np.zeros(cfg.N)
    grad_true_norm = np.zeros(cfg.N)
    
    x_val_np = cfg.y_init.cpu().numpy().flatten()
    x_tensor = cfg.y_init.clone().to(cfg.device).unsqueeze(0).requires_grad_(True)
    
    for i in range(cfg.N):
        # 1. Exact
        deterministic_val = float(x_val_np.T @ P_exact[i] @ x_val_np)
        V_true_vals[i] = deterministic_val + noise_penalty[i]
        
        grad_true_i = 2.0 * (P_exact[i] @ x_val_np)
        grad_true_norm[i] = np.linalg.norm(grad_true_i)

        # 2. NN
        t_tensor = torch.tensor([[t_grid[i]]], dtype=torch.float32, device=cfg.device)
        V_nn_i = u_theta(t_tensor, x_tensor)
        V_nn_vals[i] = V_nn_i.item()
        
        grad_nn_i = torch.autograd.grad(
            outputs=V_nn_i, 
            inputs=x_tensor,
            grad_outputs=torch.ones_like(V_nn_i),
            create_graph=False
        )[0]
        grad_nn_norm[i] = torch.norm(grad_nn_i).item()
        
    # ---------------------------------------------------------
    # C. EVALUATE ALONG THE TRAJECTORY X_t (Plot 3)
    # ---------------------------------------------------------

    V_nn_trajectory = np.zeros(cfg.N)
    V_true_trajectory = np.zeros(cfg.N)
    state_dim = x_val_np.shape[0]  # Dynamically get state dimension (e.g., 2)
    
    for i in range(cfg.N):
        # 1. Cleanly extract states and force shape to (batch_size, state_dim)
        X_tensor = X_nn[i].view(-1, state_dim)
        X_t = X_tensor.detach().cpu().numpy() 
        actual_batch = X_t.shape[0] # Handle any batch size dynamically
        
        # 2. Exact Cost along X_t
        # X_t is (batch, 2), P_exact[i] is (2, 2) -> safe to multiply
        cost_batch = np.sum((X_t @ P_exact[i]) * X_t, axis=1) + noise_penalty[i]
        V_true_trajectory[i] = cost_batch.mean()
        
        # 3. NN Cost along X_t
        t_tensor = torch.full((actual_batch, 1), t_grid[i], dtype=torch.float32, device=cfg.device)
        V_nn_i = u_theta(t_tensor, X_tensor)
        V_nn_trajectory[i] = V_nn_i.mean().item()

    # ---------------------------------------------------------
    # D. GENERATE ALL COMPARISON PLOTS
    # ---------------------------------------------------------
    clean_x = [int(val) for val in x_val_np]
    
    # PLOT 1: VALUE FUNCTION COMPARISON
    plt.figure(1, figsize=(7, 6))
    plt.plot(t_grid, V_nn_vals, color='#2ca02c', linewidth=2.5, label="PINN Value")
    plt.plot(t_grid, V_true_vals, color='red', linestyle='--', linewidth=2.5, label="Exact Theoretical Value")
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Value V(t,x)", fontsize=12)
    plt.title(f"Value Function Comparison at x = {clean_x}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path_value = "figures/exp2_value_comparison.png"
    plt.savefig(save_path_value, dpi=300, bbox_inches='tight')
    print(f"Success! Value comparison plot saved to: {save_path_value}")

    # PLOT 2: GRADIENT NORM (POLICY) COMPARISON
    plt.figure(2, figsize=(7, 6))
    plt.plot(t_grid, grad_nn_norm, color='#1f77b4', linewidth=2.5, label="PINN Gradient Norm")
    plt.plot(t_grid, grad_true_norm, color='red', linestyle='--', linewidth=2.5, label="Exact Theoretical Gradient")
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel(r"$|| \nabla_x V ||$", fontsize=14)
    plt.title(f"Policy (Gradient) Comparison at x = {clean_x}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path_policy = "figures/exp2_policy_comparison.png"
    plt.savefig(save_path_policy, dpi=300, bbox_inches='tight')
    print(f"Success! Policy comparison plot saved to: {save_path_policy}")
    
    # PLOT 3: TRAJECTORY COMPARISON
    plt.figure(3, figsize=(8, 6))
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
    
    # Show all three plots at the end!
    plt.show()

if __name__ == "__main__":
    main()
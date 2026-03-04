import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the 'soc' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soc.lqr import solve_dre
from soc.simulate import simulate_lqr

def plot_combined_convergence(N_values, errs_euler, errs_rk2, errs_rk4):
    """Plots Euler, RK2, and RK4 convergence on the same log-log plot."""
    plt.figure(figsize=(10, 6))

    # Plot actual errors
    plt.loglog(N_values, errs_euler, marker='o', label="Euler Error", color='blue', markersize=6)
    plt.loglog(N_values, errs_rk2, marker='s', label="RK2 Error", color='orange', markersize=6)
    plt.loglog(N_values, errs_rk4, marker='^', label="RK4 Error", color='purple', markersize=6)

    # Add theoretical slopes for reference
    ref_euler = [errs_euler[0] * (N_values[0]/n)**1 for n in N_values]
    ref_rk2   = [errs_rk2[0]   * (N_values[0]/n)**2 for n in N_values]
    ref_rk4   = [errs_rk4[0]   * (N_values[0]/n)**4 for n in N_values]

    plt.loglog(N_values, ref_euler, linestyle='--', color='blue', alpha=0.4, label=r"$\mathcal{O}(\Delta t)$")
    plt.loglog(N_values, ref_rk2, linestyle='--', color='orange', alpha=0.4, label=r"$\mathcal{O}(\Delta t^2)$")
    plt.loglog(N_values, ref_rk4, linestyle='--', color='purple', alpha=0.4, label=r"$\mathcal{O}(\Delta t^4)$")

    plt.title("LQR Integration Convergence: Euler vs RK2 vs RK4")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Absolute Error: |Simulated Cost - Analytic Cost|")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Save the figure to the figures/ folder
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/exp0_combined_convergence.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'figures/exp0_combined_convergence.png'")
    plt.show()

if __name__ == "__main__":
    # Problem Parameters
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    S = np.eye(2)
    y = np.array([1.0, 1.0])
    T = 1.0

    N_vals = [20, 50, 100, 200, 500, 1000, 2000 , 6000]
    errs_euler, errs_rk2, errs_rk4 = [], [], []

    print("Running combined convergence tests...")
    for N in N_vals:
        # --- EULER ---
        P_eul = solve_dre(A, B, Q, R, S, N, T, method='euler')
        _, J_eul = simulate_lqr(A, B, Q, R, S, P_eul, y, N, T, method='euler')
        errs_euler.append(abs(J_eul[-1] - (y.T @ P_eul[0] @ y)))

        # --- RK2 ---
        P_rk2 = solve_dre(A, B, Q, R, S, N, T, method='rk2')
        _, J_rk2 = simulate_lqr(A, B, Q, R, S, P_rk2, y, N, T, method='rk2')
        errs_rk2.append(abs(J_rk2[-1] - (y.T @ P_rk2[0] @ y)))

        # --- RK4 ---
        # Note: Your simulate_lqr RK4 logic requires P to have midpoints (length 2N + 1)
        # Therefore, we must solve DRE with 2*N steps to generate those midpoints!
        P_rk4 = solve_dre(A, B, Q, R, S, 2 * N, T, method='rk4')
        _, J_rk4 = simulate_lqr(A, B, Q, R, S, P_rk4, y, N, T, method='rk4')
        errs_rk4.append(abs(J_rk4[-1] - (y.T @ P_rk4[0] @ y)))

        print(f"N={N:<4} | Euler: {errs_euler[-1]:.2e} | RK2: {errs_rk2[-1]:.2e} | RK4: {errs_rk4[-1]:.2e}")

    plot_combined_convergence(N_vals, errs_euler, errs_rk2, errs_rk4)
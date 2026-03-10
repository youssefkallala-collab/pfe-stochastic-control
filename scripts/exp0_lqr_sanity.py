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

    plt.loglog(N_values, errs_euler, marker='o', label="Euler Error", color='blue', markersize=6)
    plt.loglog(N_values, errs_rk2, marker='s', label="RK2 Error", color='orange', markersize=6)
    plt.loglog(N_values, errs_rk4, marker='^', label="RK4 Error", color='purple', markersize=6)

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
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/exp0_combined_convergence.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'figures/exp0_combined_convergence.png'")
    plt.show()


def plot_cost_evolution(J_euler, J_rk2, J_rk4, T, N):
    """
    Plots the evolution of the cumulative cost J(t) over time for each method,
    and annotates the final cost value on the plot.

    Parameters
    ----------
    J_euler : array-like  Cumulative cost trajectory from Euler simulation.
    J_rk2   : array-like  Cumulative cost trajectory from RK2 simulation.
    J_rk4   : array-like  Cumulative cost trajectory from RK4 simulation.
    T       : float        Total time horizon.
    N       : int          Number of steps used (determines time grid).
    """
    t_euler = np.linspace(0, T, len(J_euler))
    t_rk2   = np.linspace(0, T, len(J_rk2))
    t_rk4   = np.linspace(0, T, len(J_rk4))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_euler, J_euler, color='blue',   label='Euler', linewidth=2)
    ax.plot(t_rk2,   J_rk2,   color='orange', label='RK2',   linewidth=2)
    ax.plot(t_rk4,   J_rk4,   color='purple', label='RK4',   linewidth=2)

    # Annotate final cost values with a vertical dashed drop-line and label
    for J_traj, color, label in [
        (J_euler, 'blue',   'Euler'),
        (J_rk2,   'orange', 'RK2'),
        (J_rk4,   'purple', 'RK4'),
    ]:
        final_cost = J_traj[-1]
        ax.annotate(
            f'{label} final: {final_cost:.6f}',
            xy=(T, final_cost),
            xytext=(-120, -20 if label != 'RK4' else -40),
            textcoords='offset points',
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, alpha=0.8),
        )
        # Dashed horizontal reference line at final value
        ax.axhline(final_cost, color=color, linestyle=':', linewidth=1, alpha=0.4)

    ax.set_title(f"Cost Evolution Over Time (N={N})")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Cumulative Cost $J(t)$")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    os.makedirs('figures', exist_ok=True)
    save_path = f'figures/exp0_cost_evolution_N{N}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cost evolution plot saved to '{save_path}'")
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

    N_vals = [20, 50, 100, 200, 500, 1000, 2000, 6000]
    errs_euler, errs_rk2, errs_rk4 = [], [], []

    # Store trajectories for the last (finest) N to plot cost evolution
    last_J_euler = last_J_rk2 = last_J_rk4 = None

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
        P_rk4 = solve_dre(A, B, Q, R, S, 2 * N, T, method='rk4')
        _, J_rk4 = simulate_lqr(A, B, Q, R, S, P_rk4, y, N, T, method='rk4')
        errs_rk4.append(abs(J_rk4[-1] - (y.T @ P_rk4[0] @ y)))

        print(f"N={N:<4} | Euler: {errs_euler[-1]:.2e} | RK2: {errs_rk2[-1]:.2e} | RK4: {errs_rk4[-1]:.2e}")

        # Keep trajectories from the finest grid for cost evolution plot
        last_J_euler, last_J_rk2, last_J_rk4 = J_eul, J_rk2, J_rk4

    plot_combined_convergence(N_vals, errs_euler, errs_rk2, errs_rk4)
    plot_cost_evolution(last_J_euler, last_J_rk2, last_J_rk4, T, N_vals[-1])
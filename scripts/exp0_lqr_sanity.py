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

    plt.loglog(N_values, errs_euler, marker='o', label="Euler Error",  color='blue',   markersize=6)
    plt.loglog(N_values, errs_rk2,   marker='s', label="RK2 Error",    color='orange', markersize=6)
    plt.loglog(N_values, errs_rk4,   marker='^', label="RK4 Error",    color='purple', markersize=6)

    ref_euler = [errs_euler[0] * (N_values[0] / n) ** 1 for n in N_values]
    ref_rk2   = [errs_rk2[0]   * (N_values[0] / n) ** 2 for n in N_values]
    ref_rk4   = [errs_rk4[0]   * (N_values[0] / n) ** 4 for n in N_values]

    plt.loglog(N_values, ref_euler, linestyle='--', color='blue',   alpha=0.4, label=r"$\mathcal{O}(\Delta t)$")
    plt.loglog(N_values, ref_rk2,   linestyle='--', color='orange', alpha=0.4, label=r"$\mathcal{O}(\Delta t^2)$")
    plt.loglog(N_values, ref_rk4,   linestyle='--', color='purple', alpha=0.4, label=r"$\mathcal{O}(\Delta t^4)$")

    plt.title("LQR Integration Convergence: Euler vs RK2 vs RK4")
    plt.xlabel("Number of Steps (N)")
    plt.ylabel("Absolute Error: |Simulated Cost - Analytic Cost|")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/exp0_combined_convergence.png', dpi=300, bbox_inches='tight')
    print("Plot saved to 'figures/exp0_combined_convergence.png'")
    plt.show()


def plot_cost_evolution(trajs_coarse, trajs_fine, analytic_cost, T, N_coarse, N_fine):
    """
    Side-by-side subplots showing cost evolution at a coarse and a fine grid.

    Left panel  (coarse N): differences between methods are clearly visible.
    Right panel (fine N):   all methods have converged; analytic cost value annotated.

    Parameters
    ----------
    trajs_coarse  : dict  {'euler': array, 'rk2': array, 'rk4': array}
    trajs_fine    : dict  {'euler': array, 'rk2': array, 'rk4': array}
    analytic_cost : float  True cost y^T P(0) y from the finest DRE solve.
    T             : float  Total time horizon.
    N_coarse      : int    Number of steps used for the coarse trajectories.
    N_fine        : int    Number of steps used for the fine trajectories.
    """
    method_styles = {
        'euler': ('Euler', 'blue',   '-'),
        'rk2':   ('RK2',   'orange', '--'),
        'rk4':   ('RK4',   'purple', ':'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, trajs, N, title, show_analytic_value in [
        (axes[0], trajs_coarse, N_coarse, f"Coarse Grid  (N = {N_coarse})", False),
        (axes[1], trajs_fine,   N_fine,   f"Fine Grid  (N = {N_fine})",     True),
    ]:
        for key, (label, color, ls) in method_styles.items():
            J = trajs[key]
            t = np.linspace(0, T, len(J))
            ax.plot(t, J, color=color, linestyle=ls, linewidth=2,
                    label=f"{label}  [final: {J[-1]:.5f}]")

        # Red horizontal line for the analytic cost — show exact value only on fine plot
        analytic_label = (f"Analytic cost: {analytic_cost:.5f}"
                          if show_analytic_value else "Analytic cost")
        ax.axhline(analytic_cost, color='red', linestyle='--', linewidth=1.5,
                   label=analytic_label)

        ax.set_title(title)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Cumulative Cost $J(t)$")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(title="Method  [final cost]", fontsize=9)

    fig.suptitle("Cost Evolution: Coarse vs Fine Integration Grid",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/exp0_cost_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Cost evolution plot saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    # ── Problem Parameters ────────────────────────────────────────────────────
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    S = np.eye(2)
    y = np.array([1.0, 1.0])
    T = 1.0

    N_vals   = [20, 50, 100, 200, 500, 1000, 2000, 6000]
    N_coarse = 20          # coarse grid where method differences are visible

    errs_euler, errs_rk2, errs_rk4 = [], [], []
    trajs_coarse: dict = {}
    trajs_fine:   dict = {}

    print("Running combined convergence tests...")
    for N in N_vals:
        # ── Euler ─────────────────────────────────────────────────────────────
        P_eul = solve_dre(A, B, Q, R, S, N, T, method='euler')
        _, J_eul = simulate_lqr(A, B, Q, R, S, P_eul, y, N, T, method='euler')
        errs_euler.append(abs(J_eul[-1] - (y.T @ P_eul[0] @ y)))

        # ── RK2 ──────────────────────────────────────────────────────────────
        P_rk2 = solve_dre(A, B, Q, R, S, N, T, method='rk2')
        _, J_rk2 = simulate_lqr(A, B, Q, R, S, P_rk2, y, N, T, method='rk2')
        errs_rk2.append(abs(J_rk2[-1] - (y.T @ P_rk2[0] @ y)))

        # ── RK4 ──────────────────────────────────────────────────────────────
        P_rk4 = solve_dre(A, B, Q, R, S, 2 * N, T, method='rk4')
        _, J_rk4 = simulate_lqr(A, B, Q, R, S, P_rk4, y, N, T, method='rk4')
        errs_rk4.append(abs(J_rk4[-1] - (y.T @ P_rk4[0] @ y)))

        print(f"N={N:<4} | Euler: {errs_euler[-1]:.2e} "
              f"| RK2: {errs_rk2[-1]:.2e} | RK4: {errs_rk4[-1]:.2e}")

        # Save coarse trajectories once, keep overwriting fine with latest
        if N == N_coarse:
            trajs_coarse = {'euler': J_eul, 'rk2': J_rk2, 'rk4': J_rk4}
        trajs_fine = {'euler': J_eul, 'rk2': J_rk2, 'rk4': J_rk4}

    # Analytic cost: y^T P(0) y using the finest RK4 solve as reference
    P_ref = solve_dre(A, B, Q, R, S, 2 * N_vals[-1], T, method='rk4')
    analytic_cost = float(y.T @ P_ref[0] @ y)
    print(f"\nAnalytic cost (reference): {analytic_cost:.6f}")

    plot_combined_convergence(N_vals, errs_euler, errs_rk2, errs_rk4)
    plot_cost_evolution(trajs_coarse, trajs_fine, analytic_cost, T, N_coarse, N_vals[-1])
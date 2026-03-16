import sys
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Ensure the 'soc' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soc.lqr import solve_dre
from soc.lqg import compute_lqg_analytic_cost
from soc.simulate import simulate_lqg_euler_maruyama, compute_lqg_mc_costs
from soc.repro import save_experiment_metadata  # <--- REPRODUCIBILITY IMPORT

# ==========================================
# 1. SETUP PARAMETERS & REPRODUCIBILITY
# ==========================================
# REQUIRED BY SECTION 8.3: Fix the random seed!
SEED = 42
np.random.seed(SEED)

d, r = 2, 2
A = np.array([[0.0, 1.0],[0.0, 0.0]])
B = np.array([[0.0], [1.0]])
Q, S = np.eye(2), np.eye(2)
R = np.array([[1.0]])
Sigma = np.array([[0.1, 0.0], [0.0, 0.1]])
y = np.array([1.0, 1.0])
N, T = 200, 1.0

M_values  =[50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
N_values  =[10, 20, 50, 100, 200, 500, 1000, 2000, 5000]  # varies for plot 3
M_fixed   = 5000   # fixed M for the N-convergence experiment
n_repeats = 200
confidence = 0.95

print(f"Running Experiment 1: Stochastic LQG Benchmark...")
print(f"Calculating stats with {n_repeats} repeats per M/N value...\n")

# ==========================================
# 2. EXACT MATHEMATICAL BASELINES (for fixed N)
# ==========================================
P = solve_dre(A, B, Q, R, S, N, T, 'euler')
analytic_cost = compute_lqg_analytic_cost(P, y, Sigma, N, T)

# ==========================================
# 3a. MC CONVERGENCE IN M  (fixed N=200)
# ==========================================
alpha  = 1.0 - confidence
t_crit = stats.t.ppf(1 - alpha / 2, df=n_repeats - 1)

means, ci_lo, ci_hi = [], [],[]

for M_val in M_values:
    run_errors =[]
    for _ in range(n_repeats):
        X_mc  = simulate_lqg_euler_maruyama(A, B, R, P, y, Sigma, N, T, M_val)
        costs = compute_lqg_mc_costs(X_mc, P, Q, R, S, B, N, T)
        run_errors.append(abs(np.mean(costs) - analytic_cost))

    mean_err = np.mean(run_errors)
    se       = np.std(run_errors, ddof=1) / np.sqrt(n_repeats)
    lo       = max(mean_err - t_crit * se, 1e-12)
    hi       = mean_err + t_crit * se

    means.append(mean_err)
    ci_lo.append(lo)
    ci_hi.append(hi)
    print(f"M={M_val:<6} | Mean Error: {mean_err:.4e} | 95% CI: [{lo:.4e}, {hi:.4e}]")

# ==========================================
# 3b. DISCRETIZATION CONVERGENCE IN N (fixed M)
# ==========================================
# For each N we need a fresh P and a reference cost at that N,
# then we compare against the finest-grid reference (N_ref).
N_ref      = 50000                                          # "ground truth" time grid
P_ref      = solve_dre(A, B, Q, R, S, N_ref, T, 'euler')
cost_ref   = compute_lqg_analytic_cost(P_ref, y, Sigma, N_ref, T)

print(f"\nRunning N-convergence experiment (M={M_fixed} fixed)...\n")

means_N, ci_lo_N, ci_hi_N = [], [],[]

for N_val in N_values:
    P_n    = solve_dre(A, B, Q, R, S, N_val, T, 'euler')

    run_errors_N =[]
    for _ in range(n_repeats):
        X_mc  = simulate_lqg_euler_maruyama(A, B, R, P_n, y, Sigma, N_val, T, M_fixed)
        costs = compute_lqg_mc_costs(X_mc, P_n, Q, R, S, B, N_val, T)
        run_errors_N.append(abs(np.mean(costs) - cost_ref))

    mean_err = np.mean(run_errors_N)
    se       = np.std(run_errors_N, ddof=1) / np.sqrt(n_repeats)
    lo       = max(mean_err - t_crit * se, 1e-12)
    hi       = mean_err + t_crit * se

    means_N.append(mean_err)
    ci_lo_N.append(lo)
    ci_hi_N.append(hi)
    print(f"N={N_val:<6} | Mean Error: {mean_err:.4e} | 95% CI:[{lo:.4e}, {hi:.4e}]")

# ==========================================
# 4. VISUALIZATION (3 plots)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

# --- Plot 1: SDE Trajectories ---
X_plot     = simulate_lqg_euler_maruyama(A, B, R, P, y, Sigma, N, T, M=50)
time_steps = np.linspace(0, T, N + 1)

for m in range(50):
    ax1.plot(time_steps, X_plot[:, m, 0], color='blue', alpha=0.15)
ax1.plot(time_steps, np.mean(X_plot[:, :, 0], axis=1), color='red', lw=2, label="MC Mean Trajectory")
ax1.set_title("LQG Euler-Maruyama (50 Trajectories)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("State $x_1$")
ax1.grid(True)
ax1.legend()

# --- Plot 2: MC Convergence in M ---
ax2.fill_between(M_values, ci_lo, ci_hi, color='steelblue', alpha=0.20,
                 label=f"{int(confidence*100)}% Confidence Interval")
ax2.loglog(M_values, means,  marker='o', color='steelblue', linewidth=2, markersize=7, label="Mean MC Error")
ax2.loglog(M_values, ci_lo,  linestyle=':', color='steelblue', linewidth=1, alpha=0.6)
ax2.loglog(M_values, ci_hi,  linestyle=':', color='steelblue', linewidth=1, alpha=0.6)

ref_M = [means[0] * np.sqrt(M_values[0] / m) for m in M_values]
ax2.loglog(M_values, ref_M, linestyle='--', color='gray', alpha=0.7, label=r"$\mathcal{O}(1/\sqrt{M})$")

ax2.annotate(f"Final mean:\n{means[-1]:.4e}",
             xy=(M_values[-1], means[-1]), xytext=(-110, 30), textcoords='offset points',
             fontsize=9, color='steelblue',
             arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='steelblue', alpha=0.85))

ax2.set_title(f"MC Convergence: Error vs $M$  ({n_repeats} repeats, N={N})")
ax2.set_xlabel("Number of Trajectories $M$")
ax2.set_ylabel(r"$|\hat{J}_M - J^*|$")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend(fontsize=9)

# --- Plot 3: Discretization Convergence in N ---
dt_values = [T / N_val for N_val in N_values]   # x-axis: step size Δt

ax3.fill_between(dt_values, ci_lo_N, ci_hi_N, color='darkorange', alpha=0.20,
                 label=f"{int(confidence*100)}% Confidence Interval")
ax3.loglog(dt_values, means_N,  marker='s', color='darkorange', linewidth=2, markersize=7, label="Mean Discretization Error")
ax3.loglog(dt_values, ci_lo_N,  linestyle=':', color='darkorange', linewidth=1, alpha=0.6)
ax3.loglog(dt_values, ci_hi_N,  linestyle=':', color='darkorange', linewidth=1, alpha=0.6)

# Euler-Maruyama has weak order 1 so the bias scales as O(Δt)
ref_N = [means_N[0] * (dt / dt_values[0]) for dt in dt_values]
ax3.loglog(dt_values, ref_N, linestyle='--', color='gray', alpha=0.7, label=r"$\mathcal{O}(\Delta t)$")

ax3.annotate(f"Final mean:\n{means_N[-1]:.4e}",
             xy=(dt_values[-1], means_N[-1]), xytext=(20, 30), textcoords='offset points',
             fontsize=9, color='darkorange',
             arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='darkorange', alpha=0.85))

ax3.set_title(f"Discretization Convergence: Error vs $\\Delta t$  (M={M_fixed})")
ax3.set_xlabel(r"Time Step $\Delta t = T/N$")
ax3.set_ylabel(r"$|\hat{J}_N - J^*_{\mathrm{ref}}|$")
ax3.grid(True, which="both", ls="--", alpha=0.5)
ax3.legend(fontsize=9)

plt.tight_layout()

# ==========================================
# 5. ABSOLUTE PATH SAVING & REPRODUCIBILITY
# ==========================================
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 1. Save the Figure
figures_folder = os.path.join(project_root, 'figures')
os.makedirs(figures_folder, exist_ok=True)
save_path = os.path.join(figures_folder, 'exp1_lqg_benchmark.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n Plot successfully saved to: {save_path}")

# 2. Save the Metadata (Git Hash, Seed, Parameters)
results_folder = os.path.join(project_root, 'results')
metadata = {
    "experiment": "Exp1_LQG_Benchmark_Advanced",
    "RNG_seed": SEED,
    "T": T,
    "N_ref_ground_truth": N_ref,
    "M_fixed_for_N_plot": M_fixed,
    "M_values_tested": M_values,
    "N_values_tested": N_values,
    "n_repeats_for_CI": n_repeats
}
save_experiment_metadata(results_folder, "exp1_lqg", metadata)

plt.show()
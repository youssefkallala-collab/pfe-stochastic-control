import pytest
import numpy as np

from soc.lqr import solve_dre
from soc.lqg import compute_lqg_analytic_cost
from soc.simulate import simulate_lqg_euler_maruyama, compute_lqg_mc_costs

# ==========================================
# 1. FIXTURES (The Pytest way to setup data)
# ==========================================
@pytest.fixture
def lqg_params():
    """Provides standard LQG matrices and parameters to any test."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    S = np.eye(2)
    Sigma = np.array([[0.1, 0.0],[0.0, 0.1]])
    y = np.array([1.0, 1.0])
    N, T = 200, 1.0
    M = 10000  # High M to reduce statistical Monte Carlo variance
    return A, B, Q, R, S, Sigma, y, N, T, M


# ==========================================
# 2. TESTS
# ==========================================
def test_lqg_cost_matches_value(lqg_params):
    """
    Test required by Section 8.2: 
    Verifies that the Monte Carlo Euler-Maruyama simulation matches the 
    analytic LQG cost (deterministic base + noise penalty).
    """
    A, B, Q, R, S, Sigma, y, N, T, M = lqg_params
    
    # Required by Section 8.3 (Reproducibility checklist):
    # Fix the random seed so this test NEVER randomly fails on your professor's computer!
    np.random.seed(42)
    
    # 1. Solve the deterministic part (Certainty Equivalence)
    P = solve_dre(A, B, Q, R, S, N, T, method='euler')
    
    # 2. Simulate the stochastic forward pass
    X = simulate_lqg_euler_maruyama(A, B, R, P, y, Sigma, N, T, M)
    
    # 3. Calculate both costs
    mc_costs = compute_lqg_mc_costs(X, P, Q, R, S, B, N, T)
    mc_cost_mean = np.mean(mc_costs)
    analytic_cost = compute_lqg_analytic_cost(P, y, Sigma, N, T)
    
    # 4. Success Criteria (Section 7.3: Target < 1% cost gap)
    cost_gap = abs(mc_cost_mean - analytic_cost) / analytic_cost
    
    assert cost_gap < 0.01, \
        f"LQG Gap too large! MC: {mc_cost_mean:.4f}, Analytic: {analytic_cost:.4f}, Gap: {cost_gap*100:.2f}%"
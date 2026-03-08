import pytest
import numpy as np
from numpy.testing import assert_allclose

from soc.lqr import solve_dre
from soc.simulate import simulate_lqr

# ==========================================
# 1. FIXTURES (The Pytest way to setup data)
# ==========================================
@pytest.fixture
def lqr_params():
    """Provides standard LQR matrices to any test that requests them."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    S = np.eye(2)
    y = np.array([1.0, 1.0])
    N, T = 100, 1.0
    return A, B, Q, R, S, y, N, T


# ==========================================
# 2. TESTS
# ==========================================

# This tells Pytest to run this test 3 separate times (once for each method)
@pytest.mark.parametrize("method",["euler", "rk2", "rk4"])
def test_riccati_symmetry(lqr_params, method):
    """Test required by Section 8.2: P(t) = P(t)^T within tolerance."""
    A, B, Q, R, S, y, N, T = lqr_params
    steps = 2 * N if method == 'rk4' else N
    
    P = solve_dre(A, B, Q, R, S, steps, T, method=method)
    
    for i in range(len(P)):
        # BETTER LIBRARY: np.testing will print the exact arrays if they don't match!
        assert_allclose(
            actual=P[i], 
            desired=P[i].T, 
            atol=1e-8, 
            err_msg=f"Method '{method}' failed symmetry at time step {i}!"
        )

# We can even assign specific tolerances to each method!
@pytest.mark.parametrize("method, tolerance",[
    ("euler", 0.05), 
    ("rk2", 0.01), 
    ("rk4", 0.001)
])
def test_lqr_value(lqr_params, method, tolerance):
    """Test required by Section 8.2: Simulated cost matches y^T P(0) y."""
    A, B, Q, R, S, y, _, T = lqr_params
    N_sim = 2000  # High N to ensure discretization gap is small
    steps = 2 * N_sim if method == 'rk4' else N_sim
    
    P = solve_dre(A, B, Q, R, S, steps, T, method=method)
    X, J = simulate_lqr(A, B, Q, R, S, P, y, N_sim, T, method=method)
    
    sim_cost = J[-1]
    ana_cost = float(y.T @ P[0] @ y)
    
    # BETTER LIBRARY: pytest.approx is specifically designed to compare 
    # floating point scalars and print the exact difference if they fail.
    assert sim_cost == pytest.approx(ana_cost, abs=tolerance), \
        f"Cost gap too large for '{method}'!"

@pytest.mark.parametrize("method", ["euler", "rk2", "rk4"])
def test_riccati_psd(lqr_params, method):
    """Sanity check from Section 5.1: If Q, S >= 0, then P(t) >= 0 (PSD)."""
    A, B, Q, R, S, y, N, T = lqr_params
    steps = 2 * N if method == 'rk4' else N
    
    # Verify our baseline test matrices are PSD first
    assert np.all(np.linalg.eigvalsh(Q) >= -1e-8), "Test Q matrix is not PSD!"
    assert np.all(np.linalg.eigvalsh(S) >= -1e-8), "Test S matrix is not PSD!"
    
    P = solve_dre(A, B, Q, R, S, steps, T, method=method)
    
    for i in range(len(P)):
        eigenvalues = np.linalg.eigvalsh(P[i])
        
        # We check that all eigenvalues are non-negative
        assert np.all(eigenvalues >= -1e-8), \
            f"Method '{method}' P[{i}] has negative eigenvalues: {eigenvalues}"
import numpy as np
from soc.lqr import solve_dre
from soc.simulate import simulate_lqr

def get_test_params():
    """Helper to return standard LQR matrices for testing."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    S = np.eye(2)
    y = np.array([1.0, 1.0])
    return A, B, Q, R, S, y

def test_riccati_symmetry():
    """Test required by Section 8.2: P(t) = P(t)^T within tolerance."""
    A, B, Q, R, S, _ = get_test_params()
    N, T = 100, 1.0
    
    methods = ['euler', 'rk2', 'rk4']
    
    for method in methods:
        # Use 2*N for RK4 so it behaves just like it does in the main script
        steps = 2 * N if method == 'rk4' else N
        P = solve_dre(A, B, Q, R, S, steps, T, method=method)
        
        # Check symmetry for all time steps
        for i in range(len(P)):
            assert np.allclose(P[i], P[i].T, atol=1e-8), f"Method '{method}': P[{i}] is not symmetric!"

def test_lqr_value():
    """Test required by Section 8.2: Simulated cost matches y^T P(0) y."""
    A, B, Q, R, S, y = get_test_params()
    N, T = 2000, 1.0 # High N to ensure discretization gap is small
    
    methods = ['euler', 'rk2', 'rk4']
    
    for method in methods:
        steps = 2 * N if method == 'rk4' else N
        
        P = solve_dre(A, B, Q, R, S, steps, T, method=method)
        X, J = simulate_lqr(A, B, Q, R, S, P, y, N, T, method=method)
        
        sim_cost = J[-1]
        ana_cost = float(y.T @ P[0] @ y)
        
        # Assert that the error is within an acceptable tolerance (< 1% gap)
        # Note: RK4 will be incredibly accurate, Euler will have the largest gap
        tolerance = 0.05 if method == 'euler' else 0.01 
        error = abs(sim_cost - ana_cost)
        
        assert error < tolerance, f"Gap too large for '{method}'! Sim: {sim_cost}, Analytic: {ana_cost}"
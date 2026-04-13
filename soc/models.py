import torch
import numpy as np

class BenchmarkConfig:
    """
    Configuration for the fully observed LQG benchmark problem.
    Stores system dynamics, cost matrices, and NN training hyperparameters.
    """
    def __init__(self):
        # ==========================================
        # 1. HARDWARE & SETUP
        # ==========================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = 2
        self.control_dim = 1
        self.x_range = 2.0
        self.y_init = torch.tensor([1.0, 1.0], dtype=torch.float32)
        # ==========================================
        # 2. TRAINING HYPERPARAMETERS
        # ==========================================
        self.batch_size = 128
        self.epochs = 3000      # Adjust based on your convergence needs
        self.lr = 1e-3          # Learning rate for Adam
        
        # ==========================================
        # 3. TIME DOMAIN
        # ==========================================
        self.T = 1.0
        self.N = 100
        self.dt = self.T / self.N
        self.M = 2*1024
        # ==========================================
        # 4. SYSTEM DYNAMICS (A, B)
        # ==========================================
        self.A = torch.tensor([[0.0, 1.0], 
                               [0.0, 0.0]], dtype=torch.float32, device=self.device)
                               
        self.B = torch.tensor([[0.0], 
                               [1.0]], dtype=torch.float32, device=self.device)
        
        # ==========================================
        # 5. NOISE / DIFFUSION (Sigma)
        # ==========================================
        sigma_val = 0.1
        self.Sigma = torch.tensor([[sigma_val, 0.0], 
                                   [0.0, sigma_val]], dtype=torch.float32, device=self.device)
        
        # ==========================================
        # 6. COST MATRICES (Q, R, S)
        # ==========================================
        self.Q = torch.eye(self.state_dim, dtype=torch.float32, device=self.device)
        self.R = torch.tensor([[1.0]], dtype=torch.float32, device=self.device)
        self.S = torch.eye(self.state_dim, dtype=torch.float32, device=self.device) # Terminal cost
        
        # ==========================================
        # 7. SIMULATION BOUNDARIES
        # ==========================================
        self.y = torch.tensor([1.0, 1.0], dtype=torch.float32, device=self.device) # Initial state
        
    def get_numpy_matrices(self):
        """
        Returns the core matrices as NumPy arrays. 
        Useful when you need to pass these to your analytic Riccati solver.
        """
        return (
            self.A.cpu().numpy(),
            self.B.cpu().numpy(),
            self.Q.cpu().numpy(),
            self.R.cpu().numpy(),
            self.S.cpu().numpy(),
            self.Sigma.cpu().numpy()
        )
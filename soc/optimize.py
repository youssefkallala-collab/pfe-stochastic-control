import torch
import torch.nn as nn
import torch.optim as optim

from soc.hamiltonian import compute_hamiltonian_partials
from soc.potential import compute_partials
from soc.hamiltonian import compute_hjb_residual

def generate_collocation_data(cfg):
    """Generates random points in space and time to train the HJB equation."""
    # Random time points between 0 and T
    t_colloc = torch.rand(cfg.batch_size, 1, device=cfg.device) * cfg.T
    
    # Random states bounded between [-cfg.x_range, cfg.x_range]
    x_colloc = (torch.rand(cfg.batch_size, cfg.state_dim, 1, device=cfg.device) * 2 - 1) * cfg.x_range
    
    return t_colloc, x_colloc

def compute_terminal_loss(u_theta, cfg):
    """Computes the loss at t=T: V(T, x) should exactly equal the terminal cost x^T S x."""
    # Fix time to the final step (T)
    t_term = torch.full((cfg.batch_size, 1), cfg.T, device=cfg.device, requires_grad=True)
    
    # Generate random states to test the boundary condition
    x_term = (torch.rand(cfg.batch_size, cfg.state_dim, 1, device=cfg.device) * 2 - 1) * cfg.x_range
    x_term.requires_grad_(True)
    
    # 1. What the network predicts at the terminal time
    V_pred = u_theta(t_term, x_term).squeeze()
    
    # 2. What the mathematical terminal cost actually is (x^T S x)
    S_batched = cfg.S.expand(cfg.batch_size, -1, -1)
    V_true = torch.bmm(torch.bmm(x_term.transpose(1, 2), S_batched), x_term).squeeze()
    
    # Penalize the difference
    return nn.MSELoss()(V_pred, V_true)

def train_potential_network(u_theta, cfg):
    """
    Main training loop for the Value Network.
    Minimizes the HJB residual in the interior and matches terminal conditions at T.
    """
    optimizer = optim.Adam(u_theta.parameters(), lr=cfg.lr)
    loss_history = []
    
    print(f"Starting Mesh-Free HJB Training for {cfg.epochs} epochs on {cfg.device}...")
    
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        
        # --- 1. INTERIOR HJB LOSS ---
        t_colloc, x_colloc = generate_collocation_data(cfg)
        
        # Extract derivatives using your mesh-free autodiff wrapper
        V, V_t, V_x, Hessian = compute_partials(u_theta, t_colloc, x_colloc)
        
        # Calculate how far we are from V_t + H* = 0
        hjb_residual = compute_hjb_residual(cfg, x_colloc, V_t, V_x, Hessian)       
         
        # The ideal residual is 0 everywhere, so we minimize MSE against zeros
        
        loss_hjb = nn.MSELoss()(hjb_residual, torch.zeros_like(hjb_residual))
        
        # --- 2. TERMINAL LOSS ---
        loss_terminal = compute_terminal_loss(u_theta, cfg)
        
        # --- 3. TOTAL LOSS & BACKPROP ---
        loss = loss_hjb + loss_terminal
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 50 == 0 or epoch == cfg.epochs - 1:
            u_theta.eval() # Temporarily put network in eval mode
            with torch.no_grad():
                # Define the test points: [1.0, 1.0], [0.5, 0.5], and [-1.0, -1.0]
                x_tests = torch.tensor(
                    [[1.0, 1.0], [0.5, 0.5], [-1.0, -1.0]],
                    dtype=torch.float32, device=cfg.device
                )
                # Evaluate at time t = 0
                t_test = torch.zeros(len(x_tests), 1, device=cfg.device)
                
                # Forward pass to get V(0, x)
                preds = u_theta(t_test, x_tests)
            
            u_theta.train() # Put network back in training mode

            # Print the detailed metrics using YOUR variable names
            print(
                f"Epoch {epoch:04d}/{cfg.epochs} | "
                f"HJB: {loss_hjb.item():.6f} | "
                f"Term: {loss_terminal.item():.6f} | "
                f"Total: {loss.item():.6f} | "
                f"V(0,[1,1])={preds[0].item():.4f}  "
                f"V(0,[.5,.5])={preds[1].item():.4f}  "
                f"V(0,[-1,-1])={preds[2].item():.4f}"
            )
            
    print("Training Complete!")
    return loss_history
import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """
    Neural Network approximating the optimal Cost-to-Go / Value Function V(t, x).
    """
    def __init__(self, cfg):
        super(ValueNetwork, self).__init__()
        
        # Input dimension: time (1D) + state (state_dim)
        input_dim = 1 + cfg.state_dim
        hidden_dim = 64*2*2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # CRITICAL: We must use a smooth activation like SiLU (Swish) or Tanh.
            # ReLU cannot be used because its second derivative is zero everywhere, 
            # which would destroy the diffusion/Hessian term in the HJB equation!
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, t, x):
        """
        Args:
            t: Time tensor of shape (batch_size, 1)
            x: State tensor of shape (batch_size, state_dim, 1)
        Returns:
            V: Value tensor of shape (batch_size, 1)
        """
        # Flatten x from (batch_size, state_dim, 1) to (batch_size, state_dim)
        x_flat = x.squeeze(-1) 
        
        # Concatenate time and state inputs
        inputs = torch.cat([t, x_flat], dim=1)
        return self.net(inputs)


def compute_partials(u_theta, t, x):
    """
    Computes the Value, its time derivative, spatial gradient, and spatial Hessian.
    Uses PyTorch Autograd to extract exact derivatives without mesh grids.
    """
    # 1. Flag inputs to track operations for gradient calculation
    t.requires_grad_(True)
    x.requires_grad_(True)
    
    batch_size, state_dim, _ = x.shape
    
    # 2. Forward Pass: Compute V(t, x)
    V = u_theta(t, x)
    
    # 3. First Derivatives (Gradient w.r.t t and x)
    # create_graph=True is MANDATORY because we need to differentiate V_x AGAIN to get the Hessian.
    grad_outputs = torch.ones_like(V)
    V_t, V_x = torch.autograd.grad(
        outputs=V, 
        inputs=(t, x), 
        grad_outputs=grad_outputs, 
        create_graph=True
    )
    
    # 4. Second Derivative (Hessian w.r.t x)
    Hessian = torch.zeros(batch_size, state_dim, state_dim, device=x.device)
    
    # Compute the Hessian row by row across the batch
    for i in range(state_dim):
        # Differentiate the i-th component of V_x with respect to x
        grad_V_x_i = torch.autograd.grad(
            outputs=V_x[:, i, :], 
            inputs=x, 
            grad_outputs=torch.ones_like(V_x[:, i, :]), 
            create_graph=True,
            retain_graph=True
        )[0]
        Hessian[:, i, :] = grad_V_x_i.squeeze(-1)
        
    return V, V_t, V_x, Hessian
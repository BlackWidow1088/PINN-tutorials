import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import os
import re
from typing import Tuple, Dict
import matplotlib.pyplot as plt

# Set device and precision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class OpenFOAMDataLoader:
    """Load and parse OpenFOAM data files"""
    
    def __init__(self, case_dir: str):
        self.case_dir = case_dir
        self.nu = 1e-6  # Kinematic viscosity from physicalProperties
        self.rho = 1000.0  # Water density (kg/m³)
        
    def read_points(self) -> np.ndarray:
        """Read mesh points from constant/polyMesh/points"""
        points_file = os.path.join(self.case_dir, 'constant/polyMesh/points')
        with open(points_file, 'r') as f:
            content = f.read()
        
        # Extract number of points
        match = re.search(r'(\d+)\s*\(', content)
        n_points = int(match.group(1))
        
        # Extract coordinates
        pattern = r'\(([-\d\.eE\+\-]+)\s+([-\d\.eE\+\-]+)\s+([-\d\.eE\+\-]+)\)'
        matches = re.findall(pattern, content)
        
        points = np.array([[float(x), float(y), float(z)] for x, y, z in matches])
        return points
    
    def read_scalar_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """Read scalar field (e.g., pressure)"""
        field_file = os.path.join(self.case_dir, time_dir, field_name)
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Find internalField section
        match = re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)', content)
        if not match:
            return None
        
        n_cells = int(match.group(1))
        
        # Extract values
        pattern = r'([-\d\.eE\+\-]+)'
        matches = re.findall(pattern, content)
        
        # Skip header values, get only internal field
        start_idx = content.find('(')
        values_str = content[start_idx:]
        values = re.findall(r'([-\d\.eE\+\-]+)', values_str)
        
        # Take first n_cells values
        field = np.array([float(v) for v in values[:n_cells]], dtype=np.float64)
        return field
    
    def read_vector_field(self, time_dir: str, field_name: str) -> np.ndarray:
        """Read vector field (e.g., velocity)"""
        field_file = os.path.join(self.case_dir, time_dir, field_name)
        with open(field_file, 'r') as f:
            content = f.read()
        
        # Find internalField section
        match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)', content)
        if not match:
            return None
        
        n_cells = int(match.group(1))
        
        # Extract vector values
        pattern = r'\(([-\d\.eE\+\-]+)\s+([-\d\.eE\+\-]+)\s+([-\d\.eE\+\-]+)\)'
        matches = re.findall(pattern, content)
        
        # Take first n_cells vectors
        vectors = np.array([[float(x), float(y), float(z)] 
                           for x, y, z in matches[:n_cells]], dtype=np.float64)
        return vectors
    
    def get_cell_centers(self, points: np.ndarray) -> np.ndarray:
        """Estimate cell centers from points (simplified - in practice use OpenFOAM tools)"""
        # For simplicity, we'll use mesh points as collocation points
        # In practice, you'd compute actual cell centers
        return points
    
    def load_simulation_data(self, time_dirs: list = None) -> Dict:
        """Load all simulation data"""
        if time_dirs is None:
            time_dirs = ['100', '200', '300', '400', '500', 
                        '600', '700', '800', '900', '1000']
        
        points = self.read_points()
        
        # Load data from one time step (steady state, so all should be similar)
        # Use the most converged solution (1000)
        time_dir = '1000'
        
        p = self.read_scalar_field(time_dir, 'p')
        U = self.read_vector_field(time_dir, 'U')
        
        # Get cell centers (using points for now)
        # Note: In practice, you need to map cell centers properly
        cell_centers = self.get_cell_centers(points)
        
        # Match the number of points to the number of internal field cells
        # Field data has values for internal cells only (typically fewer than total mesh points)
        n_cells = len(p) if p is not None else len(U) if U is not None else len(cell_centers)
        
        # Use first n_cells points to match field data size
        # In practice, you'd want to properly map cell centers to field data
        if n_cells < len(cell_centers):
            # Use first n_cells points
            valid_indices = np.arange(n_cells)
        else:
            # If we have more cells than points (unlikely), use all points
            valid_indices = np.arange(len(cell_centers))
            # Truncate field data to match
            if p is not None and len(p) > len(cell_centers):
                p = p[:len(cell_centers)]
            if U is not None and len(U) > len(cell_centers):
                U = U[:len(cell_centers)]
        
        
        return {
            'points': cell_centers[valid_indices],
            'pressure': p,
            'velocity': U,
            'nu': self.nu,
            'rho': self.rho
        }


class PINN(nn.Module):
    """Physics-Informed Neural Network for Navier-Stokes"""
    
    def __init__(self, input_dim=3, hidden_layers=[50, 50, 50, 50], output_dim=4):
        """
        Args:
            input_dim: 3 for (x, y, z)
            output_dim: 4 for (u, v, w, p)
        """
        super(PINN, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())  # Tanh works well for PINNs
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass: (x, y, z) -> (u, v, w, p)"""
        return self.network(x)


class PhysicsInformedLoss:
    """Compute physics-informed loss for Navier-Stokes"""
    
    def __init__(self, nu: float, rho: float, device: torch.device):
        self.nu = nu
        self.rho = rho
        self.device = device
    
    def compute_derivatives(self, u, x, create_graph=True):
        """Compute first and second derivatives using automatic differentiation"""
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=create_graph, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                   create_graph=create_graph, retain_graph=True)[0]
        
        return u_x, u_xx
    
    def compute_residuals(self, x, u_pred, v_pred, w_pred, p_pred):
        """
        Compute Navier-Stokes residuals
        Equations:
        r1: Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        r2: x-momentum: u·∇u + ∂p/∂x/ρ - ν∇²u = 0
        r3: y-momentum: v·∇v + ∂p/∂y/ρ - ν∇²v = 0
        r4: z-momentum: w·∇w + ∂p/∂z/ρ - ν∇²w = 0
        """
        x.requires_grad_(True)
        
        # First derivatives
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred),
                                 create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred),
                                 create_graph=True, retain_graph=True)[0]
        w_x = torch.autograd.grad(w_pred, x, grad_outputs=torch.ones_like(w_pred),
                                 create_graph=True, retain_graph=True)[0]
        p_x = torch.autograd.grad(p_pred, x, grad_outputs=torch.ones_like(p_pred),
                                 create_graph=True, retain_graph=True)[0]
        
        # Extract partial derivatives
        u_x_val = u_x[:, 0]
        u_y_val = u_x[:, 1]
        u_z_val = u_x[:, 2]
        
        v_x_val = v_x[:, 0]
        v_y_val = v_x[:, 1]
        v_z_val = v_x[:, 2]
        
        w_x_val = w_x[:, 0]
        w_y_val = w_x[:, 1]
        w_z_val = w_x[:, 2]
        
        p_x_val = p_x[:, 0]
        p_y_val = p_x[:, 1]
        p_z_val = p_x[:, 2]
        
        # Second derivatives for Laplacian
        u_xx = torch.autograd.grad(u_x_val, x, grad_outputs=torch.ones_like(u_x_val),
                                   create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y_val, x, grad_outputs=torch.ones_like(u_y_val),
                                   create_graph=True, retain_graph=True)[0]
        u_zz = torch.autograd.grad(u_z_val, x, grad_outputs=torch.ones_like(u_z_val),
                                   create_graph=True, retain_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x_val, x, grad_outputs=torch.ones_like(v_x_val),
                                   create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y_val, x, grad_outputs=torch.ones_like(v_y_val),
                                   create_graph=True, retain_graph=True)[0]
        v_zz = torch.autograd.grad(v_z_val, x, grad_outputs=torch.ones_like(v_z_val),
                                   create_graph=True, retain_graph=True)[0]
        
        w_xx = torch.autograd.grad(w_x_val, x, grad_outputs=torch.ones_like(w_x_val),
                                   create_graph=True, retain_graph=True)[0]
        w_yy = torch.autograd.grad(w_y_val, x, grad_outputs=torch.ones_like(w_y_val),
                                   create_graph=True, retain_graph=True)[0]
        w_zz = torch.autograd.grad(w_z_val, x, grad_outputs=torch.ones_like(w_z_val),
                                   create_graph=True, retain_graph=True)[0]
        
        # Laplacians
        laplacian_u = u_xx[:, 0] + u_yy[:, 1] + u_zz[:, 2]
        laplacian_v = v_xx[:, 0] + v_yy[:, 1] + v_zz[:, 2]
        laplacian_w = w_xx[:, 0] + w_yy[:, 1] + w_zz[:, 2]
        
        # Residual 1: Continuity equation
        r1 = u_x_val + v_y_val + w_z_val
        
        # Residual 2: x-momentum
        r2 = (u_pred * u_x_val + v_pred * u_y_val + w_pred * u_z_val +
              p_x_val / self.rho - self.nu * laplacian_u)
        
        # Residual 3: y-momentum
        r3 = (u_pred * v_x_val + v_pred * v_y_val + w_pred * v_z_val +
              p_y_val / self.rho - self.nu * laplacian_v)
        
        # Residual 4: z-momentum
        r4 = (u_pred * w_x_val + v_pred * w_y_val + w_pred * w_z_val +
              p_z_val / self.rho - self.nu * laplacian_w)
        
        return r1, r2, r3, r4
    
    def compute_loss(self, x_colloc, u_pred, v_pred, w_pred, p_pred,
                    x_data=None, u_data=None, v_data=None, w_data=None, p_data=None,
                    u_pred_data=None, v_pred_data=None, w_pred_data=None, p_pred_data=None,
                    lambda_data=1.0, lambda_physics=1.0):
        """
        Compute total loss: L = λ_data * L_data + λ_physics * L_physics
        where L_physics = MSE(r1) + MSE(r2) + MSE(r3) + MSE(r4)
        """
        
        # Physics loss (collocation points)
        r1, r2, r3, r4 = self.compute_residuals(x_colloc, u_pred, v_pred, w_pred, p_pred)
        
        loss_physics = (torch.mean(r1**2) + torch.mean(r2**2) + 
                       torch.mean(r3**2) + torch.mean(r4**2))
        
        # Data loss (if provided)
        loss_data = torch.tensor(0.0, device=self.device)
        if x_data is not None and u_data is not None and u_pred_data is not None:
            if u_data is not None and u_pred_data is not None:
                loss_data += torch.mean((u_pred_data - u_data)**2)
            if v_data is not None and v_pred_data is not None:
                loss_data += torch.mean((v_pred_data - v_data)**2)
            if w_data is not None and w_pred_data is not None:
                loss_data += torch.mean((w_pred_data - w_data)**2)
            if p_data is not None and p_pred_data is not None:
                loss_data += torch.mean((p_pred_data - p_data)**2)
        
        total_loss = lambda_data * loss_data + lambda_physics * loss_physics
        
        return total_loss, loss_physics, loss_data


class DataNormalizer:
    """Normalize input and output data to [0, 1] range"""
    
    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.u_min = None
        self.u_max = None
        self.v_min = None
        self.v_max = None
        self.w_min = None
        self.w_max = None
        self.p_min = None
        self.p_max = None
    
    def fit(self, x, u, v, w, p):
        """Compute normalization parameters"""
        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)
        
        self.u_min = u.min()
        self.u_max = u.max()
        self.v_min = v.min()
        self.v_max = v.max()
        self.w_min = w.min()
        self.w_max = w.max()
        self.p_min = p.min()
        self.p_max = p.max()
    
    def normalize_x(self, x):
        """Normalize spatial coordinates"""
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
    
    def normalize_u(self, u):
        """Normalize u velocity"""
        return (u - self.u_min) / (self.u_max - self.u_min + 1e-8)
    
    def normalize_v(self, v):
        """Normalize v velocity"""
        return (v - self.v_min) / (self.v_max - self.v_min + 1e-8)
    
    def normalize_w(self, w):
        """Normalize w velocity"""
        return (w - self.w_min) / (self.w_max - self.w_min + 1e-8)
    
    def normalize_p(self, p):
        """Normalize pressure"""
        return (p - self.p_min) / (self.p_max - self.p_min + 1e-8)
    
    def denormalize_u(self, u_norm):
        """Denormalize u velocity"""
        return u_norm * (self.u_max - self.u_min) + self.u_min
    
    def denormalize_v(self, v_norm):
        """Denormalize v velocity"""
        return v_norm * (self.v_max - self.v_min) + self.v_min
    
    def denormalize_w(self, w_norm):
        """Denormalize w velocity"""
        return w_norm * (self.w_max - self.w_min) + self.w_min
    
    def denormalize_p(self, p_norm):
        """Denormalize pressure"""
        return p_norm * (self.p_max - self.p_min) + self.p_min


def sample_collocation_points(points, n_points=10000, wall_weight=2.0):
    """
    Sample collocation points with more density near walls
    where gradients are large
    """
    # Compute distance from center (pipe radius = 1)
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    
    # Weight points near wall (r close to 1)
    weights = 1.0 + wall_weight * np.exp(-10 * (1.0 - r)**2)
    weights = weights / weights.sum()
    
    # Sample points
    indices = np.random.choice(len(points), size=n_points, p=weights, replace=False)
    return points[indices]


def train_pinn(case_dir: str, n_epochs_adam=10000, n_epochs_lbfgs=1000,
               n_colloc=10000, lr=1e-3, lambda_data=1.0, lambda_physics=1.0):
    """Main training function"""
    
    print("Loading OpenFOAM data...")
    loader = OpenFOAMDataLoader(case_dir)
    data = loader.load_simulation_data()
    
    # Extract data
    points = data['points']
    p_data = data['pressure']
    U_data = data['velocity']
    u_data = U_data[:, 0]
    v_data = U_data[:, 1]
    w_data = U_data[:, 2]
    nu = data['nu']
    rho = data['rho']
    
    
    print(f"Loaded {len(points)} points")
    print(f"Pressure range: [{p_data.min():.2e}, {p_data.max():.2e}]")
    print(f"Velocity range: [{U_data.min():.2e}, {U_data.max():.2e}]")
    
    # Normalize data
    print("Normalizing data...")
    normalizer = DataNormalizer()
    normalizer.fit(points, u_data, v_data, w_data, p_data)
    
    x_norm = normalizer.normalize_x(points)
    u_norm = normalizer.normalize_u(u_data)
    v_norm = normalizer.normalize_v(v_data)
    w_norm = normalizer.normalize_w(w_data)
    p_norm = normalizer.normalize_p(p_data)
    
    # Sample collocation points
    print(f"Sampling {n_colloc} collocation points...")
    x_colloc = sample_collocation_points(points, n_points=n_colloc)
    x_colloc_norm = normalizer.normalize_x(x_colloc)
    
    # Convert to tensors
    
    x_colloc_tensor = torch.tensor(x_colloc_norm, dtype=torch.float64, device=device, requires_grad=True)
    x_data_tensor = torch.tensor(x_norm, dtype=torch.float64, device=device)
    u_data_tensor = torch.tensor(u_norm, dtype=torch.float64, device=device)
    v_data_tensor = torch.tensor(v_norm, dtype=torch.float64, device=device)
    w_data_tensor = torch.tensor(w_norm, dtype=torch.float64, device=device)
    p_data_tensor = torch.tensor(p_norm, dtype=torch.float64, device=device)
    
    
    # Initialize model
    model = PINN(input_dim=3, hidden_layers=[50, 50, 50, 50], output_dim=4).to(device)
    physics_loss = PhysicsInformedLoss(nu, rho, device)
    
    # Adam optimizer
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)
    
    print("\nTraining with Adam optimizer...")
    losses = []
    for epoch in range(n_epochs_adam):
        optimizer_adam.zero_grad()
        
        # Forward pass on collocation points
        output_colloc = model(x_colloc_tensor)
        u_pred_colloc = output_colloc[:, 0]
        v_pred_colloc = output_colloc[:, 1]
        w_pred_colloc = output_colloc[:, 2]
        p_pred_colloc = output_colloc[:, 3]
        
        # Forward pass on data points
        output_data = model(x_data_tensor)
        u_pred_data = output_data[:, 0]
        v_pred_data = output_data[:, 1]
        w_pred_data = output_data[:, 2]
        p_pred_data = output_data[:, 3]
        
        # Compute loss
        
        loss, loss_physics, loss_data = physics_loss.compute_loss(
            x_colloc_tensor, u_pred_colloc, v_pred_colloc, w_pred_colloc, p_pred_colloc,
            x_data_tensor, u_data_tensor, v_data_tensor, w_data_tensor, p_data_tensor,
            u_pred_data=u_pred_data, v_pred_data=v_pred_data, w_pred_data=w_pred_data, p_pred_data=p_pred_data,
            lambda_data=lambda_data, lambda_physics=lambda_physics
        )
        
        loss.backward()
        optimizer_adam.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{n_epochs_adam}, Loss: {loss.item():.6e}, "
                  f"Physics: {loss_physics.item():.6e}, Data: {loss_data.item():.6e}")
    
    # L-BFGS optimizer (optional fine-tuning)
    if n_epochs_lbfgs > 0:
        print("\nFine-tuning with L-BFGS optimizer...")
        optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=20, 
                                      history_size=50, tolerance_grad=1e-5)
        
        def closure():
            optimizer_lbfgs.zero_grad()
            output_colloc = model(x_colloc_tensor)
            u_pred_colloc = output_colloc[:, 0]
            v_pred_colloc = output_colloc[:, 1]
            w_pred_colloc = output_colloc[:, 2]
            p_pred_colloc = output_colloc[:, 3]
            
            output_data = model(x_data_tensor)
            u_pred_data = output_data[:, 0]
            v_pred_data = output_data[:, 1]
            w_pred_data = output_data[:, 2]
            p_pred_data = output_data[:, 3]
            
            loss, _, _ = physics_loss.compute_loss(
                x_colloc_tensor, u_pred_colloc, v_pred_colloc, w_pred_colloc, p_pred_colloc,
                x_data_tensor, u_data_tensor, v_data_tensor, w_data_tensor, p_data_tensor,
                u_pred_data=u_pred_data, v_pred_data=v_pred_data, w_pred_data=w_pred_data, p_pred_data=p_pred_data,
                lambda_data=lambda_data, lambda_physics=lambda_physics
            )
            loss.backward()
            return loss
        
        for epoch in range(n_epochs_lbfgs):
            loss = optimizer_lbfgs.step(closure)
            if (epoch + 1) % 100 == 0:
                print(f"L-BFGS Epoch {epoch+1}/{n_epochs_lbfgs}, Loss: {loss.item():.6e}")
    
    return model, normalizer, losses


if __name__ == "__main__":
    case_dir = "/Users/abhijeetchavan/Desktop/cylinderCase"
    
    # Training parameters
    n_epochs_adam = 10000
    n_epochs_lbfgs = 1000
    n_colloc = 10000  # Number of collocation points
    lr = 1e-3
    lambda_data = 1.0
    lambda_physics = 1.0
    
    # Train model
    model, normalizer, losses = train_pinn(
        case_dir, 
        n_epochs_adam=n_epochs_adam,
        n_epochs_lbfgs=n_epochs_lbfgs,
        n_colloc=n_colloc,
        lr=lr,
        lambda_data=lambda_data,
        lambda_physics=lambda_physics
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer': normalizer,
    }, 'pinn_model.pt')
    
    print("\nTraining complete! Model saved to pinn_model.pt")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Training loss plot saved to training_loss.png")

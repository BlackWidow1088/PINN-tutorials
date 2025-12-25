"""
Verification utilities for PINN model
Provides functions for model loading, data loading, coordinate evaluation, and residual computation
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, Optional
import os
import re
from pinn import OpenFOAMDataLoader, PINN, DataNormalizer, PhysicsInformedLoss


def load_trained_model(model_path: str, device: torch.device) -> Tuple[nn.Module, DataNormalizer]:
    """
    Load model and normalizer from checkpoint
    
    Args:
        model_path: Path to saved model file (pinn_model.pt)
        device: Torch device (cpu or cuda)
        
    Returns:
        model: Loaded PINN model
        normalizer: DataNormalizer instance with fitted parameters
    """
    # #region agent log
    import json
    import numpy as np
    with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"A","location":"verify_pinn.py:28","message":"load_trained_model entry","data":{"model_path":model_path,"device":str(device),"torch_version":torch.__version__,"numpy_version":np.__version__,"has_numpy_core":hasattr(np, 'core'),"has_numpy__core":hasattr(np, '_core')},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion agent log
    
    # #region agent log
    import json
    with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"B","location":"verify_pinn.py:29","message":"Before torch.load","data":{"weights_only_default":True if hasattr(torch.load, '__defaults__') else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion agent log
    
    try:
        # Use weights_only=False for trusted local files (PyTorch 2.6+ default changed)
        # Also ensure DataNormalizer is available in __main__ for unpickling
        import sys
        import __main__
        # #region agent log
        import json
        with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"H","location":"verify_pinn.py:42","message":"Before DataNormalizer injection","data":{"has_dataclass_in_main":hasattr(__main__, 'DataNormalizer'),"main_module":str(__main__)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion agent log
        if not hasattr(__main__, 'DataNormalizer'):
            __main__.DataNormalizer = DataNormalizer
            # #region agent log
            import json
            with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"I","location":"verify_pinn.py:47","message":"DataNormalizer injected into __main__","data":{"injected":True},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion agent log
        
        # Load checkpoint with numpy version compatibility
        # Safely map numpy._core to numpy.core in sys.modules only (no attribute modification)
        import numpy as np
        import sys
        # #region agent log
        import json
        with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"K","location":"verify_pinn.py:59","message":"Before torch.load","data":{"numpy_version":np.__version__,"has_numpy_core":hasattr(np, 'core'),"has_numpy__core":hasattr(np, '_core'),"numpy__core_in_sys_modules":"numpy._core" in sys.modules},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion agent log
        
        # Only modify sys.modules - do NOT touch numpy attributes to avoid segfaults
        if not hasattr(np, '_core') and 'numpy._core' not in sys.modules:
            import numpy.core
            sys.modules['numpy._core'] = numpy.core
            # #region agent log
            import json
            with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"Q","location":"verify_pinn.py:70","message":"Mapped numpy._core to numpy.core in sys.modules","data":{"numpy__core_in_sys_modules":"numpy._core" in sys.modules,"numpy_core_type":type(sys.modules.get('numpy._core')).__name__ if 'numpy._core' in sys.modules else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion agent log
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # #region agent log
            import json
            with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"J","location":"verify_pinn.py:102","message":"torch.load succeeded","data":{"checkpoint_keys":list(checkpoint.keys()) if checkpoint else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion agent log
        except (SystemError, ModuleNotFoundError, AttributeError) as e:
            # #region agent log
            import json
            with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"P","location":"verify_pinn.py:108","message":"torch.load failed with numpy version mismatch","data":{"error_type":type(e).__name__,"error_message":str(e)[:300]},"timestamp":int(__import__('time').time()*1000)})+'\n')
            # #endregion agent log
            # Re-raise with helpful message
            raise RuntimeError(
                f"Failed to load model due to numpy version mismatch. "
                f"Model was saved with numpy 2.0+ (uses numpy._core) but current version is {np.__version__} (uses numpy.core). "
                f"Please either: (1) Upgrade numpy to 2.0+, or (2) Re-save the model with the current numpy version. "
                f"Original error: {str(e)}"
            ) from e
    except Exception as e:
        # #region agent log
        import json
        with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"D","location":"verify_pinn.py:31","message":"torch.load failed","data":{"error_type":type(e).__name__,"error_message":str(e)},"timestamp":int(__import__('time').time()*1000)})+'\n')
        # #endregion agent log
        raise
    
    # #region agent log
    import json
    with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"E","location":"verify_pinn.py:33","message":"Before model reconstruction","data":{"has_model_state_dict":"model_state_dict" in checkpoint,"has_normalizer":"normalizer" in checkpoint,"normalizer_type":type(checkpoint.get('normalizer')).__name__ if 'normalizer' in checkpoint else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion agent log
    
    # Reconstruct model architecture (same as training)
    model = PINN(input_dim=3, hidden_layers=[50, 50, 50, 50], output_dim=4).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # #region agent log
    import json
    with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"F","location":"verify_pinn.py:40","message":"Before normalizer access","data":{"normalizer_in_checkpoint":"normalizer" in checkpoint},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion agent log
    
    # Get normalizer
    normalizer = checkpoint['normalizer']
    
    # #region agent log
    import json
    with open('/Users/abhijeetchavan/Desktop/cylinderCase/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"pre-fix","hypothesisId":"G","location":"verify_pinn.py:44","message":"load_trained_model exit","data":{"normalizer_type":type(normalizer).__name__ if normalizer else None},"timestamp":int(__import__('time').time()*1000)})+'\n')
    # #endregion agent log
    
    return model, normalizer


def load_simulation_data_for_time(case_dir: str, time_dir: str) -> Dict:
    """
    Load simulation data from a specific time directory
    
    Args:
        case_dir: Path to OpenFOAM case directory
        time_dir: Time directory name (e.g., '100', '500', '1000')
        
    Returns:
        Dictionary containing points, pressure, velocity, nu, rho
    """
    loader = OpenFOAMDataLoader(case_dir)
    
    # Read points
    points = loader.read_points()
    
    # Read fields from specified time directory
    p = loader.read_scalar_field(time_dir, 'p')
    U = loader.read_vector_field(time_dir, 'U')
    
    # Get cell centers (using points for now)
    cell_centers = loader.get_cell_centers(points)
    
    # Match the number of points to the number of internal field cells
    n_cells = len(p) if p is not None else len(U) if U is not None else len(cell_centers)
    
    if n_cells < len(cell_centers):
        valid_indices = np.arange(n_cells)
    else:
        valid_indices = np.arange(len(cell_centers))
        if p is not None and len(p) > len(cell_centers):
            p = p[:len(cell_centers)]
        if U is not None and len(U) > len(cell_centers):
            U = U[:len(cell_centers)]
    
    return {
        'points': cell_centers[valid_indices],
        'pressure': p,
        'velocity': U,
        'nu': loader.nu,
        'rho': loader.rho
    }


def evaluate_pinn_at_point(model: nn.Module, normalizer: DataNormalizer, 
                          x: float, y: float, z: float, 
                          nu: float, rho: float, device: torch.device) -> Dict:
    """
    Evaluate PINN at specific coordinate, return predictions and residuals
    
    Args:
        model: Trained PINN model
        normalizer: DataNormalizer instance
        x, y, z: Spatial coordinates
        nu: Kinematic viscosity
        rho: Density
        device: Torch device
        
    Returns:
        Dictionary with predictions (u, v, w, p) and residuals (r1, r2, r3, r4)
    """
    # Normalize input coordinates
    coords = np.array([[x, y, z]], dtype=np.float64)
    coords_norm = normalizer.normalize_x(coords)
    
    # Convert to tensor
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float64, device=device, requires_grad=True)
    
    # Forward pass
    with torch.no_grad():
        output = model(coords_tensor)
    
    # Extract predictions (normalized)
    u_pred_norm = output[0, 0].item()
    v_pred_norm = output[0, 1].item()
    w_pred_norm = output[0, 2].item()
    p_pred_norm = output[0, 3].item()
    
    # Denormalize
    u_pred = normalizer.denormalize_u(u_pred_norm)
    v_pred = normalizer.denormalize_v(v_pred_norm)
    w_pred = normalizer.denormalize_w(w_pred_norm)
    p_pred = normalizer.denormalize_p(p_pred_norm)
    
    # Compute residuals (need gradients, so re-run with requires_grad)
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float64, device=device, requires_grad=True)
    output = model(coords_tensor)
    u_pred_tensor = output[:, 0]
    v_pred_tensor = output[:, 1]
    w_pred_tensor = output[:, 2]
    p_pred_tensor = output[:, 3]
    
    # Compute residuals using PhysicsInformedLoss
    physics_loss = PhysicsInformedLoss(nu, rho, device)
    r1, r2, r3, r4 = physics_loss.compute_residuals(
        coords_tensor, u_pred_tensor, v_pred_tensor, w_pred_tensor, p_pred_tensor
    )
    
    return {
        'u': u_pred,
        'v': v_pred,
        'w': w_pred,
        'p': p_pred,
        'r1': r1[0].item(),
        'r2': r2[0].item(),
        'r3': r3[0].item(),
        'r4': r4[0].item()
    }


def get_ground_truth_at_point(points: np.ndarray, u_data: np.ndarray, 
                              v_data: np.ndarray, w_data: np.ndarray, 
                              p_data: np.ndarray, x: float, y: float, z: float,
                              nu: float, rho: float, device: torch.device) -> Dict:
    """
    Get ground truth values at specific coordinate using nearest neighbor
    
    Args:
        points: Array of mesh points (N, 3)
        u_data, v_data, w_data, p_data: Ground truth field data
        x, y, z: Target coordinates
        nu: Kinematic viscosity
        rho: Density
        device: Torch device
        
    Returns:
        Dictionary with ground truth values (u, v, w, p) and residuals (r1, r2, r3, r4)
    """
    # Find nearest neighbor
    target_point = np.array([[x, y, z]])
    distances = cdist(target_point, points)
    nearest_idx = np.argmin(distances[0])
    
    # Get ground truth values
    u_gt = u_data[nearest_idx]
    v_gt = v_data[nearest_idx]
    w_gt = w_data[nearest_idx]
    p_gt = p_data[nearest_idx]
    
    # Compute residuals from ground truth (using finite differences approximation)
    # For simplicity, we'll use a small perturbation to estimate derivatives
    eps = 1e-5
    target_tensor = torch.tensor([[x, y, z]], dtype=torch.float64, device=device, requires_grad=False)
    
    # Create a simple function to compute residuals from discrete data
    # This is approximate - ideally we'd interpolate and compute exact derivatives
    # For now, we'll compute residuals using the nearest point's neighbors
    neighbors = points[np.argsort(distances[0])[:10]]  # Get 10 nearest neighbors
    
    if len(neighbors) > 1:
        # Approximate gradients using neighbors
        dx = neighbors[:, 0] - x
        dy = neighbors[:, 1] - y
        dz = neighbors[:, 2] - z
        
        # Get values at neighbors
        neighbor_distances = cdist(neighbors, points)
        neighbor_indices = np.argmin(neighbor_distances, axis=1)
        
        u_neighbors = u_data[neighbor_indices]
        v_neighbors = v_data[neighbor_indices]
        w_neighbors = w_data[neighbor_indices]
        p_neighbors = p_data[neighbor_indices]
        
        # Approximate partial derivatives (simple finite difference)
        # This is a simplified approach - in practice, use proper interpolation
        du_dx = np.mean((u_neighbors - u_gt) / (dx + 1e-10))
        du_dy = np.mean((u_neighbors - u_gt) / (dy + 1e-10))
        du_dz = np.mean((u_neighbors - u_gt) / (dz + 1e-10))
        
        dv_dx = np.mean((v_neighbors - v_gt) / (dx + 1e-10))
        dv_dy = np.mean((v_neighbors - v_gt) / (dy + 1e-10))
        dv_dz = np.mean((v_neighbors - v_gt) / (dz + 1e-10))
        
        dw_dx = np.mean((w_neighbors - w_gt) / (dx + 1e-10))
        dw_dy = np.mean((w_neighbors - w_gt) / (dy + 1e-10))
        dw_dz = np.mean((w_neighbors - w_gt) / (dz + 1e-10))
        
        dp_dx = np.mean((p_neighbors - p_gt) / (dx + 1e-10))
        dp_dy = np.mean((p_neighbors - p_gt) / (dy + 1e-10))
        dp_dz = np.mean((p_neighbors - p_gt) / (dz + 1e-10))
        
        # Approximate second derivatives (simplified)
        # For Laplacian, we need second derivatives
        # This is a rough approximation
        d2u_dx2 = np.mean((u_neighbors - u_gt) / ((dx + 1e-10)**2))
        d2u_dy2 = np.mean((u_neighbors - u_gt) / ((dy + 1e-10)**2))
        d2u_dz2 = np.mean((u_neighbors - u_gt) / ((dz + 1e-10)**2))
        
        d2v_dx2 = np.mean((v_neighbors - v_gt) / ((dx + 1e-10)**2))
        d2v_dy2 = np.mean((v_neighbors - v_gt) / ((dy + 1e-10)**2))
        d2v_dz2 = np.mean((v_neighbors - v_gt) / ((dz + 1e-10)**2))
        
        d2w_dx2 = np.mean((w_neighbors - w_gt) / ((dx + 1e-10)**2))
        d2w_dy2 = np.mean((w_neighbors - w_gt) / ((dy + 1e-10)**2))
        d2w_dz2 = np.mean((w_neighbors - w_gt) / ((dz + 1e-10)**2))
        
        laplacian_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        laplacian_v = d2v_dx2 + d2v_dy2 + d2v_dz2
        laplacian_w = d2w_dx2 + d2w_dy2 + d2w_dz2
        
        # Compute residuals
        r1 = du_dx + dv_dy + dw_dz  # Continuity
        r2 = u_gt * du_dx + v_gt * du_dy + w_gt * du_dz + dp_dx / rho - nu * laplacian_u  # x-momentum
        r3 = u_gt * dv_dx + v_gt * dv_dy + w_gt * dv_dz + dp_dy / rho - nu * laplacian_v  # y-momentum
        r4 = u_gt * dw_dx + v_gt * dw_dy + w_gt * dw_dz + dp_dz / rho - nu * laplacian_w  # z-momentum
    else:
        # Fallback: set residuals to zero if we can't compute
        r1 = r2 = r3 = r4 = 0.0
    
    return {
        'u': u_gt,
        'v': v_gt,
        'w': w_gt,
        'p': p_gt,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'r4': r4,
        'nearest_point': points[nearest_idx],
        'distance': distances[0, nearest_idx]
    }


def compute_residuals_at_point(u: float, v: float, w: float, p: float,
                               x: float, y: float, z: float,
                               nu: float, rho: float, device: torch.device,
                               model: Optional[nn.Module] = None,
                               normalizer: Optional[DataNormalizer] = None) -> Tuple[float, float, float, float]:
    """
    Compute Navier-Stokes residuals at a point
    
    Args:
        u, v, w, p: Velocity and pressure values
        x, y, z: Spatial coordinates
        nu: Kinematic viscosity
        rho: Density
        device: Torch device
        model: Optional PINN model (if provided, uses automatic differentiation)
        normalizer: Optional normalizer (if using model)
        
    Returns:
        Tuple of residuals (r1, r2, r3, r4)
    """
    if model is not None and normalizer is not None:
        # Use model's automatic differentiation
        coords = np.array([[x, y, z]], dtype=np.float64)
        coords_norm = normalizer.normalize_x(coords)
        coords_tensor = torch.tensor(coords_norm, dtype=torch.float64, device=device, requires_grad=True)
        
        output = model(coords_tensor)
        u_pred = output[:, 0]
        v_pred = output[:, 1]
        w_pred = output[:, 2]
        p_pred = output[:, 3]
        
        physics_loss = PhysicsInformedLoss(nu, rho, device)
        r1, r2, r3, r4 = physics_loss.compute_residuals(
            coords_tensor, u_pred, v_pred, w_pred, p_pred
        )
        
        return r1[0].item(), r2[0].item(), r3[0].item(), r4[0].item()
    else:
        # Simplified residual computation (would need proper gradient computation)
        # For now, return zeros as placeholder
        return 0.0, 0.0, 0.0, 0.0


def compute_residual_statistics(points: np.ndarray, model: nn.Module, 
                               normalizer: DataNormalizer, nu: float, rho: float,
                               device: torch.device, n_samples: int = 1000) -> Dict:
    """
    Compute residual statistics on a sample of collocation points
    
    Args:
        points: Mesh points
        model: Trained PINN model
        normalizer: DataNormalizer instance
        nu: Kinematic viscosity
        rho: Density
        device: Torch device
        n_samples: Number of points to sample
        
    Returns:
        Dictionary with residual statistics
    """
    # Sample points
    if n_samples > len(points):
        n_samples = len(points)
    indices = np.random.choice(len(points), size=n_samples, replace=False)
    sample_points = points[indices]
    
    # Normalize
    sample_points_norm = normalizer.normalize_x(sample_points)
    sample_tensor = torch.tensor(sample_points_norm, dtype=torch.float64, device=device, requires_grad=True)
    
    # Forward pass
    output = model(sample_tensor)
    u_pred = output[:, 0]
    v_pred = output[:, 1]
    w_pred = output[:, 2]
    p_pred = output[:, 3]
    
    # Compute residuals
    physics_loss = PhysicsInformedLoss(nu, rho, device)
    r1, r2, r3, r4 = physics_loss.compute_residuals(
        sample_tensor, u_pred, v_pred, w_pred, p_pred
    )
    
    # Convert to numpy
    r1_np = r1.detach().cpu().numpy()
    r2_np = r2.detach().cpu().numpy()
    r3_np = r3.detach().cpu().numpy()
    r4_np = r4.detach().cpu().numpy()
    
    # Compute statistics
    stats = {
        'r1': {
            'mean': float(np.mean(r1_np)),
            'std': float(np.std(r1_np)),
            'max': float(np.max(np.abs(r1_np))),
            'min': float(np.min(r1_np)),
            'l2_norm': float(np.linalg.norm(r1_np))
        },
        'r2': {
            'mean': float(np.mean(r2_np)),
            'std': float(np.std(r2_np)),
            'max': float(np.max(np.abs(r2_np))),
            'min': float(np.min(r2_np)),
            'l2_norm': float(np.linalg.norm(r2_np))
        },
        'r3': {
            'mean': float(np.mean(r3_np)),
            'std': float(np.std(r3_np)),
            'max': float(np.max(np.abs(r3_np))),
            'min': float(np.min(r3_np)),
            'l2_norm': float(np.linalg.norm(r3_np))
        },
        'r4': {
            'mean': float(np.mean(r4_np)),
            'std': float(np.std(r4_np)),
            'max': float(np.max(np.abs(r4_np))),
            'min': float(np.min(r4_np)),
            'l2_norm': float(np.linalg.norm(r4_np))
        },
        'n_samples': n_samples
    }
    
    return stats


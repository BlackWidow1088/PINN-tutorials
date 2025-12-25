"""
Verification report generator with statistics and pass/fail criteria
"""

from typing import Dict, Optional, Tuple
from datetime import datetime


def generate_verification_report(
    pinn_results: Dict,
    gt_results: Dict,
    coordinates: Tuple[float, float, float],
    residual_stats: Optional[Dict] = None,
    nu: Optional[float] = None,
    rho: Optional[float] = None
) -> str:
    """
    Generate a comprehensive verification report
    
    Args:
        pinn_results: Dictionary with PINN predictions and residuals
        gt_results: Dictionary with ground truth values and residuals
        coordinates: Tuple of (x, y, z) coordinates
        residual_stats: Optional dictionary with residual statistics
        
    Returns:
        Markdown formatted report string
    """
    x, y, z = coordinates
    
    # Compute errors
    u_error = abs(pinn_results['u'] - gt_results['u'])
    v_error = abs(pinn_results['v'] - gt_results['v'])
    w_error = abs(pinn_results['w'] - gt_results['w'])
    p_error = abs(pinn_results['p'] - gt_results['p'])
    
    u_rel_error = 100 * u_error / (abs(gt_results['u']) + 1e-10)
    v_rel_error = 100 * v_error / (abs(gt_results['v']) + 1e-10)
    w_rel_error = 100 * w_error / (abs(gt_results['w']) + 1e-10)
    p_rel_error = 100 * p_error / (abs(gt_results['p']) + 1e-10)
    
    # Pass/fail criteria
    max_rel_error = 5.0  # 5% relative error threshold
    max_abs_residual = 1e-3  # Maximum acceptable residual
    
    u_pass = u_rel_error < max_rel_error
    v_pass = v_rel_error < max_rel_error
    w_pass = w_rel_error < max_rel_error
    p_pass = p_rel_error < max_rel_error
    
    r1_pass = abs(pinn_results['r1']) < max_abs_residual
    r2_pass = abs(pinn_results['r2']) < max_abs_residual
    r3_pass = abs(pinn_results['r3']) < max_abs_residual
    r4_pass = abs(pinn_results['r4']) < max_abs_residual
    
    overall_pass = (u_pass and v_pass and w_pass and p_pass and 
                   r1_pass and r2_pass and r3_pass and r4_pass)
    
    # Generate report
    report = f"""# PINN Model Verification Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Evaluation Point

- **Coordinates:** ({x:.6f}, {y:.6f}, {z:.6f})
- **Distance from center:** {((x**2 + y**2)**0.5):.6f} m

---

## 2. Prediction Comparison

### 2.1 Velocity Components

| Variable | PINN Prediction | Ground Truth | Absolute Error | Relative Error (%) | Status |
|----------|----------------|--------------|----------------|-------------------|--------|
| u (m/s)  | {pinn_results['u']:.6e} | {gt_results['u']:.6e} | {u_error:.6e} | {u_rel_error:.4f}% | {'✅ PASS' if u_pass else '❌ FAIL'} |
| v (m/s)  | {pinn_results['v']:.6e} | {gt_results['v']:.6e} | {v_error:.6e} | {v_rel_error:.4f}% | {'✅ PASS' if v_pass else '❌ FAIL'} |
| w (m/s)  | {pinn_results['w']:.6e} | {gt_results['w']:.6e} | {w_error:.6e} | {w_rel_error:.4f}% | {'✅ PASS' if w_pass else '❌ FAIL'} |

### 2.2 Pressure

| Variable | PINN Prediction | Ground Truth | Absolute Error | Relative Error (%) | Status |
|----------|----------------|--------------|----------------|-------------------|--------|
| p (Pa)   | {pinn_results['p']:.6e} | {gt_results['p']:.6e} | {p_error:.6e} | {p_rel_error:.4f}% | {'✅ PASS' if p_pass else '❌ FAIL'} |

**Pass Criteria:** Relative error < {max_rel_error}%

---

## 3. Physics Residuals (Navier-Stokes Equations)

### 3.1 PINN Residuals

| Residual | Equation | Value | Status |
|----------|----------|-------|--------|
| r1 | Continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 | {pinn_results['r1']:.6e} | {'✅ PASS' if r1_pass else '❌ FAIL'} |
| r2 | x-momentum | {pinn_results['r2']:.6e} | {'✅ PASS' if r2_pass else '❌ FAIL'} |
| r3 | y-momentum | {pinn_results['r3']:.6e} | {'✅ PASS' if r3_pass else '❌ FAIL'} |
| r4 | z-momentum | {pinn_results['r4']:.6e} | {'✅ PASS' if r4_pass else '❌ FAIL'} |

### 3.2 Ground Truth Residuals

| Residual | Equation | Value |
|----------|----------|-------|
| r1 | Continuity | {gt_results['r1']:.6e} |
| r2 | x-momentum | {gt_results['r2']:.6e} |
| r3 | y-momentum | {gt_results['r3']:.6e} |
| r4 | z-momentum | {gt_results['r4']:.6e} |

**Pass Criteria:** |Residual| < {max_abs_residual}

---

## 4. Residual Statistics

"""
    
    if residual_stats:
        report += f"""
### 4.1 Summary Statistics

**Number of samples:** {residual_stats.get('n_samples', 'N/A')}

| Residual | Mean | Std Dev | Max |Value|| Min Value | L2 Norm |
|----------|------|---------|-----|-----|-----------|---------|
| r1 (Continuity) | {residual_stats['r1']['mean']:.6e} | {residual_stats['r1']['std']:.6e} | {residual_stats['r1']['max']:.6e} | {residual_stats['r1']['min']:.6e} | {residual_stats['r1']['l2_norm']:.6e} |
| r2 (x-momentum) | {residual_stats['r2']['mean']:.6e} | {residual_stats['r2']['std']:.6e} | {residual_stats['r2']['max']:.6e} | {residual_stats['r2']['min']:.6e} | {residual_stats['r2']['l2_norm']:.6e} |
| r3 (y-momentum) | {residual_stats['r3']['mean']:.6e} | {residual_stats['r3']['std']:.6e} | {residual_stats['r3']['max']:.6e} | {residual_stats['r3']['min']:.6e} | {residual_stats['r3']['l2_norm']:.6e} |
| r4 (z-momentum) | {residual_stats['r4']['mean']:.6e} | {residual_stats['r4']['std']:.6e} | {residual_stats['r4']['max']:.6e} | {residual_stats['r4']['min']:.6e} | {residual_stats['r4']['l2_norm']:.6e} |

### 4.2 Pass/Fail Analysis

"""
        
        # Check statistics pass/fail
        stats_pass = True
        for r in ['r1', 'r2', 'r3', 'r4']:
            mean_abs = abs(residual_stats[r]['mean'])
            max_abs = residual_stats[r]['max']
            if mean_abs > max_abs_residual or max_abs > max_abs_residual * 10:
                stats_pass = False
                break
        
        report += f"""
- **Mean residual check:** {'✅ PASS' if stats_pass else '❌ FAIL'} (Mean |residual| < {max_abs_residual})
- **Max residual check:** {'✅ PASS' if max([residual_stats[r]['max'] for r in ['r1', 'r2', 'r3', 'r4']]) < max_abs_residual * 10 else '❌ FAIL'} (Max |residual| < {max_abs_residual * 10})

"""
    else:
        report += "*Residual statistics not computed. Use the 'Compute Residual Statistics' button to generate statistics.*\n\n"
    
    # Overall assessment
    report += f"""
---

## 5. Overall Assessment

### 5.1 Test Results Summary

- **Prediction Accuracy:** {'✅ PASS' if (u_pass and v_pass and w_pass and p_pass) else '❌ FAIL'}
- **Physics Residuals:** {'✅ PASS' if (r1_pass and r2_pass and r3_pass and r4_pass) else '❌ FAIL'}
"""
    
    if residual_stats:
        stats_overall_pass = all([
            abs(residual_stats[r]['mean']) < max_abs_residual 
            for r in ['r1', 'r2', 'r3', 'r4']
        ])
        report += f"- **Residual Statistics:** {'✅ PASS' if stats_overall_pass else '❌ FAIL'}\n"
    
    report += f"""
### 5.2 Final Verdict

**{'✅ VERIFICATION PASSED' if overall_pass else '❌ VERIFICATION FAILED'}**

"""
    
    if not overall_pass:
        report += "### Issues Identified:\n\n"
        if not u_pass:
            report += f"- u velocity relative error ({u_rel_error:.4f}%) exceeds threshold ({max_rel_error}%)\n"
        if not v_pass:
            report += f"- v velocity relative error ({v_rel_error:.4f}%) exceeds threshold ({max_rel_error}%)\n"
        if not w_pass:
            report += f"- w velocity relative error ({w_rel_error:.4f}%) exceeds threshold ({max_rel_error}%)\n"
        if not p_pass:
            report += f"- pressure relative error ({p_rel_error:.4f}%) exceeds threshold ({max_rel_error}%)\n"
        if not r1_pass:
            report += f"- Continuity residual ({abs(pinn_results['r1']):.6e}) exceeds threshold ({max_abs_residual})\n"
        if not r2_pass:
            report += f"- x-momentum residual ({abs(pinn_results['r2']):.6e}) exceeds threshold ({max_abs_residual})\n"
        if not r3_pass:
            report += f"- y-momentum residual ({abs(pinn_results['r3']):.6e}) exceeds threshold ({max_abs_residual})\n"
        if not r4_pass:
            report += f"- z-momentum residual ({abs(pinn_results['r4']):.6e}) exceeds threshold ({max_abs_residual})\n"
    
    report += f"""
---

## 6. Recommendations

"""
    
    if overall_pass:
        report += "- ✅ Model performance meets verification criteria\n"
        report += "- ✅ Physics residuals are within acceptable limits\n"
        report += "- ✅ Model is ready for deployment\n"
    else:
        report += "- ⚠️ Review model training parameters\n"
        report += "- ⚠️ Consider increasing training epochs or adjusting loss weights\n"
        report += "- ⚠️ Verify data quality and normalization\n"
        report += "- ⚠️ Check boundary conditions and physics constraints\n"
    
    report += f"""
---

## 7. Technical Details

- **Model Architecture:** 3 input → [50, 50, 50, 50] hidden → 4 output
- **Activation Function:** Tanh
- **Normalization:** Min-max normalization to [0, 1]
- **Physics:** Incompressible Navier-Stokes equations
- **Viscosity (ν):** {nu if nu is not None else 'N/A'} m²/s
- **Density (ρ):** {rho if rho is not None else 'N/A'} kg/m³

---

*End of Report*
"""
    
    return report


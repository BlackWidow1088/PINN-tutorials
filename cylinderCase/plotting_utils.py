"""
Enhanced plotting utilities with clear value labels and annotations
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict


def plot_3d_pipe_with_point(x: float, y: float, z: float, 
                            pipe_radius: float = 1.0, 
                            pipe_length: float = 5.0) -> go.Figure:
    """
    Create 3D visualization of pipe with marked point
    
    Args:
        x, y, z: Coordinates of the point to mark
        pipe_radius: Radius of the pipe (default: 1.0)
        pipe_length: Length of the pipe (default: 5.0)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Create pipe geometry
    z_pipe = np.linspace(-pipe_length/2, pipe_length/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    
    # Create cylindrical surface
    z_surf, theta_surf = np.meshgrid(z_pipe, theta)
    x_surf = pipe_radius * np.cos(theta_surf)
    y_surf = pipe_radius * np.sin(theta_surf)
    
    # Add pipe surface (semi-transparent)
    fig.add_trace(go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        colorscale='Blues',
        opacity=0.3,
        showscale=False,
        name='Pipe Wall'
    ))
    
    # Add inlet circle (z = -pipe_length/2)
    theta_inlet = np.linspace(0, 2*np.pi, 100)
    x_inlet = pipe_radius * np.cos(theta_inlet)
    y_inlet = pipe_radius * np.sin(theta_inlet)
    z_inlet = np.full_like(theta_inlet, -pipe_length/2)
    
    fig.add_trace(go.Scatter3d(
        x=x_inlet, y=y_inlet, z=z_inlet,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Inlet',
        showlegend=True
    ))
    
    # Add outlet circle (z = pipe_length/2)
    z_outlet = np.full_like(theta_inlet, pipe_length/2)
    fig.add_trace(go.Scatter3d(
        x=x_inlet, y=y_inlet, z=z_outlet,
        mode='lines',
        line=dict(color='red', width=3),
        name='Outlet',
        showlegend=True
    ))
    
    # Add coordinate axes
    axis_length = pipe_length * 0.6
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=4),
        text=['', 'X'],
        textposition='top center',
        name='X-axis',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=4),
        text=['', 'Y'],
        textposition='middle right',
        name='Y-axis',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines+text',
        line=dict(color='blue', width=4),
        text=['', 'Z'],
        textposition='top center',
        name='Z-axis',
        showlegend=False
    ))
    
    # Add marked point
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers+text',
        marker=dict(
            size=12,
            color='red',
            symbol='circle',
            line=dict(color='darkred', width=2)
        ),
        text=[f'({x:.3f}, {y:.3f}, {z:.3f})'],
        textposition='top center',
        name='Selected Point',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'3D Pipe Visualization - Point at ({x:.3f}, {y:.3f}, {z:.3f})',
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def plot_residual_histograms(stats: Dict) -> go.Figure:
    """
    Plot residual distribution histograms with value labels
    
    Args:
        stats: Dictionary with residual statistics
        
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('r1: Continuity', 'r2: x-momentum', 
                       'r3: y-momentum', 'r4: z-momentum'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    residuals = ['r1', 'r2', 'r3', 'r4']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for residual, (row, col) in zip(residuals, positions):
        # Create histogram data (simulated from statistics)
        # In practice, you'd use actual residual values
        mean = stats[residual]['mean']
        std = stats[residual]['std']
        
        # Generate sample data for visualization
        n_samples = stats.get('n_samples', 1000)
        sample_data = np.random.normal(mean, std, min(1000, n_samples))
        
        # Create histogram
        fig.add_trace(
            go.Histogram(
                x=sample_data,
                nbinsx=30,
                name=residual,
                showlegend=False,
                marker=dict(
                    color='steelblue',
                    line=dict(color='black', width=1)
                ),
                hovertemplate=f'<b>{residual}</b><br>' +
                            'Value: %{x:.6e}<br>' +
                            'Count: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add statistics annotations
        fig.add_annotation(
            x=0.5, y=0.95,
            xref=f'x{residuals.index(residual) + 1} domain',
            yref=f'y{residuals.index(residual) + 1} domain',
            text=f"Mean: {mean:.4e}<br>Std: {std:.4e}",
            showarrow=False,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    
    fig.update_layout(
        title={
            'text': 'Residual Distribution Histograms',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 5):
        fig.update_xaxes(title_text="Residual Value", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
        fig.update_yaxes(title_text="Frequency", row=(i-1)//2 + 1, col=(i-1)%2 + 1)
    
    return fig


def plot_residual_statistics(stats: Dict) -> go.Figure:
    """
    Plot residual statistics bar chart with value labels
    
    Args:
        stats: Dictionary with residual statistics
        
    Returns:
        Plotly figure object
    """
    residuals = ['r1', 'r2', 'r3', 'r4']
    residual_names = ['Continuity', 'x-momentum', 'y-momentum', 'z-momentum']
    
    # Extract statistics
    means = [abs(stats[r]['mean']) for r in residuals]
    stds = [stats[r]['std'] for r in residuals]
    max_vals = [stats[r]['max'] for r in residuals]
    l2_norms = [stats[r]['l2_norm'] for r in residuals]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean |Residual|', 'Standard Deviation', 
                       'Max |Residual|', 'L2 Norm'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Mean values
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=means,
            name='Mean |Residual|',
            marker_color='steelblue',
            text=[f'{m:.4e}' for m in means],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Mean: %{y:.6e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Standard deviations
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=stds,
            name='Std Dev',
            marker_color='coral',
            text=[f'{s:.4e}' for s in stds],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Std: %{y:.6e}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Max values
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=max_vals,
            name='Max |Residual|',
            marker_color='lightgreen',
            text=[f'{m:.4e}' for m in max_vals],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>Max: %{y:.6e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # L2 norms
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=l2_norms,
            name='L2 Norm',
            marker_color='gold',
            text=[f'{l:.4e}' for l in l2_norms],
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{x}</b><br>L2: %{y:.6e}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=2)
    
    fig.update_layout(
        title={
            'text': 'Residual Statistics',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        showlegend=False
    )
    
    return fig


def plot_comparison_charts(pinn_results: Dict, gt_results: Dict) -> go.Figure:
    """
    Plot comparison charts between PINN predictions and ground truth
    
    Args:
        pinn_results: Dictionary with PINN predictions
        gt_results: Dictionary with ground truth values
        
    Returns:
        Plotly figure object
    """
    variables = ['u', 'v', 'w', 'p']
    variable_names = ['u (m/s)', 'v (m/s)', 'w (m/s)', 'p (Pa)']
    
    pinn_values = [pinn_results[v] for v in variables]
    gt_values = [gt_results[v] for v in variables]
    
    # Compute errors
    abs_errors = [abs(p - g) for p, g in zip(pinn_values, gt_values)]
    rel_errors = [100 * abs(p - g) / (abs(g) + 1e-10) for p, g in zip(pinn_values, gt_values)]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Value Comparison', 'Absolute Error', 
                       'Relative Error (%)', 'Residual Comparison'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Value comparison
    fig.add_trace(
        go.Bar(
            x=variable_names,
            y=pinn_values,
            name='PINN',
            marker_color='steelblue',
            text=[f'{v:.4e}' for v in pinn_values],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>PINN: %{y:.6e}<extra></extra>'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=variable_names,
            y=gt_values,
            name='Ground Truth',
            marker_color='coral',
            text=[f'{v:.4e}' for v in gt_values],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>GT: %{y:.6e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Absolute error
    fig.add_trace(
        go.Bar(
            x=variable_names,
            y=abs_errors,
            name='Absolute Error',
            marker_color='red',
            text=[f'{e:.4e}' for e in abs_errors],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>Error: %{y:.6e}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Relative error
    fig.add_trace(
        go.Bar(
            x=variable_names,
            y=rel_errors,
            name='Relative Error (%)',
            marker_color='orange',
            text=[f'{e:.2f}%' for e in rel_errors],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>Error: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Residual comparison
    residual_names = ['r1 (Continuity)', 'r2 (x-mom)', 'r3 (y-mom)', 'r4 (z-mom)']
    pinn_residuals = [abs(pinn_results[f'r{i+1}']) for i in range(4)]
    gt_residuals = [abs(gt_results[f'r{i+1}']) for i in range(4)]
    
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=pinn_residuals,
            name='PINN Residuals',
            marker_color='steelblue',
            text=[f'{r:.4e}' for r in pinn_residuals],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>PINN: %{y:.6e}<extra></extra>'
        ),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(
            x=residual_names,
            y=gt_residuals,
            name='GT Residuals',
            marker_color='coral',
            text=[f'{r:.4e}' for r in gt_residuals],
            textposition='outside',
            textfont=dict(size=9),
            hovertemplate='<b>%{x}</b><br>GT: %{y:.6e}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)
    fig.update_yaxes(title_text="Error (%)", row=2, col=1)
    fig.update_yaxes(title_text="|Residual|", row=2, col=2)
    
    fig.update_layout(
        title={
            'text': 'PINN vs Ground Truth Comparison',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


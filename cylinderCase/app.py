"""
Streamlit web application for PINN model verification
Provides interactive coordinate input, 3D visualization, and comparison with ground truth
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional
import os
import sys
# Import DataNormalizer to make it available in __main__ for pickle loading
from pinn import DataNormalizer
# Make DataNormalizer available in __main__ namespace for unpickling
if __name__ == '__main__' or hasattr(sys.modules.get('__main__'), '__file__'):
    import __main__
    __main__.DataNormalizer = DataNormalizer

from verify_pinn import (
    load_trained_model, 
    load_simulation_data_for_time,
    evaluate_pinn_at_point,
    get_ground_truth_at_point,
    compute_residual_statistics
)
from plotting_utils import (
    plot_3d_pipe_with_point,
    plot_residual_histograms,
    plot_residual_statistics,
    plot_comparison_charts
)
from report_generator import generate_verification_report

# Set page config
st.set_page_config(
    page_title="PINN Model Verification",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'normalizer' not in st.session_state:
    st.session_state.normalizer = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'nu' not in st.session_state:
    st.session_state.nu = None
if 'rho' not in st.session_state:
    st.session_state.rho = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Device setup
device = st.session_state.device
torch.set_default_dtype(torch.float64)

# Auto-load model and data using cached functions
@st.cache_resource
def load_model_cached(model_path: str, device: torch.device):
    """Load and cache the PINN model"""
    return load_trained_model(model_path, device)

@st.cache_data
def load_data_cached(case_dir: str, time_dir: str):
    """Load and cache simulation data"""
    return load_simulation_data_for_time(case_dir, time_dir)

# Sidebar
st.sidebar.title("⚙️ Configuration")

# Time step selector
time_steps = ['100', '200', '300', '400', '500', '600', '700', '800', '900', '1000']
selected_time = st.sidebar.selectbox(
    "Time Step",
    options=time_steps,
    index=len(time_steps) - 1,  # Default to 1000
    help="Select the time step for ground truth data"
)

# Auto-load model and data
# Use paths relative to the script location (works on Streamlit Cloud)
# Handle both local execution and Streamlit Cloud deployment
try:
    # Try to get the directory where app.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for Streamlit Cloud or when __file__ is not available
    script_dir = os.getcwd()
    
model_path = os.path.join(script_dir, "pinn_model.pt")  # Model is in same directory as app.py
case_dir = script_dir  # OpenFOAM case files are in same directory as app.py

# Main content
st.title("🔬 PINN Model Verification")
st.markdown("Interactive verification tool for Physics-Informed Neural Network predictions")

# Load model automatically (cached, only loads once)
if st.session_state.model is None:
    with st.spinner("Loading model..."):
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.error(f"Current working directory: {os.getcwd()}")
            st.error(f"Script directory: {script_dir}")
            st.error(f"Files in script directory: {', '.join(os.listdir(script_dir)[:10])}")
            st.stop()
        try:
            model, normalizer = load_model_cached(model_path, device)
            st.session_state.model = model
            st.session_state.normalizer = normalizer
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.error(f"Model path attempted: {model_path}")
            st.stop()
else:
    model = st.session_state.model
    normalizer = st.session_state.normalizer

# Load data automatically (reloads when time step changes)
# Check if we need to reload data (time step changed or not loaded)
if ('selected_time' not in st.session_state or 
    st.session_state.selected_time != selected_time or 
    st.session_state.data is None):
    with st.spinner(f"Loading simulation data from time step {selected_time}..."):
        try:
            data = load_data_cached(case_dir, selected_time)
            st.session_state.data = data
            st.session_state.nu = data['nu']
            st.session_state.rho = data['rho']
            st.session_state.selected_time = selected_time
        except Exception as e:
            st.error(f"❌ Error loading simulation data: {str(e)}")
            st.stop()
else:
    data = st.session_state.data

# Coordinate input section
st.header("📍 Coordinate Input")
col1, col2, col3 = st.columns(3)

with col1:
    x = st.number_input(
        "X coordinate",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="X coordinate in the pipe (typically -1 to 1)"
    )

with col2:
    y = st.number_input(
        "Y coordinate",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Y coordinate in the pipe (typically -1 to 1)"
    )

with col3:
    z = st.number_input(
        "Z coordinate",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Z coordinate along pipe axis (typically -2.5 to 2.5)"
    )

# Validate coordinates
pipe_radius = 1.0
pipe_length = 5.0
r = np.sqrt(x**2 + y**2)

if r > pipe_radius:
    st.warning(f"⚠️ Point is outside pipe radius! Distance from center: {r:.3f} (max: {pipe_radius})")
if abs(z) > pipe_length / 2:
    st.warning(f"⚠️ Point is outside pipe length! Z: {z:.3f} (range: [-{pipe_length/2}, {pipe_length/2}])")

# Evaluate button
if st.button("🔍 Evaluate", type="primary"):
    with st.spinner("Evaluating..."):
        try:
            # Evaluate PINN
            pinn_results = evaluate_pinn_at_point(
                st.session_state.model,
                st.session_state.normalizer,
                x, y, z,
                st.session_state.nu,
                st.session_state.rho,
                device
            )
            
            # Get ground truth
            points = st.session_state.data['points']
            u_data = st.session_state.data['velocity'][:, 0]
            v_data = st.session_state.data['velocity'][:, 1]
            w_data = st.session_state.data['velocity'][:, 2]
            p_data = st.session_state.data['pressure']
            
            gt_results = get_ground_truth_at_point(
                points, u_data, v_data, w_data, p_data,
                x, y, z,
                st.session_state.nu,
                st.session_state.rho,
                device
            )
            
            # Store results in session state
            st.session_state.pinn_results = pinn_results
            st.session_state.gt_results = gt_results
            st.session_state.evaluated_coords = (x, y, z)
            
            st.success("✅ Evaluation complete!")
            
        except Exception as e:
            st.error(f"❌ Error during evaluation: {str(e)}")
            st.stop()

# Display results if available
if 'pinn_results' in st.session_state and 'gt_results' in st.session_state:
    st.header("📊 Results")
    
    # 3D Visualization
    st.subheader("🎨 3D Pipe Visualization")
    x_coord, y_coord, z_coord = st.session_state.evaluated_coords
    fig_3d = plot_3d_pipe_with_point(x_coord, y_coord, z_coord)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Comparison table
    st.subheader("📋 Comparison Table")
    
    comparison_data = {
        'Variable': ['u (m/s)', 'v (m/s)', 'w (m/s)', 'p (Pa)'],
        'PINN Prediction': [
            f"{st.session_state.pinn_results['u']:.6e}",
            f"{st.session_state.pinn_results['v']:.6e}",
            f"{st.session_state.pinn_results['w']:.6e}",
            f"{st.session_state.pinn_results['p']:.6e}"
        ],
        'Ground Truth': [
            f"{st.session_state.gt_results['u']:.6e}",
            f"{st.session_state.gt_results['v']:.6e}",
            f"{st.session_state.gt_results['w']:.6e}",
            f"{st.session_state.gt_results['p']:.6e}"
        ],
        'Absolute Error': [
            f"{abs(st.session_state.pinn_results['u'] - st.session_state.gt_results['u']):.6e}",
            f"{abs(st.session_state.pinn_results['v'] - st.session_state.gt_results['v']):.6e}",
            f"{abs(st.session_state.pinn_results['w'] - st.session_state.gt_results['w']):.6e}",
            f"{abs(st.session_state.pinn_results['p'] - st.session_state.gt_results['p']):.6e}"
        ],
        'Relative Error (%)': [
            f"{100 * abs(st.session_state.pinn_results['u'] - st.session_state.gt_results['u']) / (abs(st.session_state.gt_results['u']) + 1e-10):.4f}",
            f"{100 * abs(st.session_state.pinn_results['v'] - st.session_state.gt_results['v']) / (abs(st.session_state.gt_results['v']) + 1e-10):.4f}",
            f"{100 * abs(st.session_state.pinn_results['w'] - st.session_state.gt_results['w']) / (abs(st.session_state.gt_results['w']) + 1e-10):.4f}",
            f"{100 * abs(st.session_state.pinn_results['p'] - st.session_state.gt_results['p']) / (abs(st.session_state.gt_results['p']) + 1e-10):.4f}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Navier-Stokes Residuals
    st.subheader("⚗️ Navier-Stokes Residuals")
    
    residual_data = {
        'Residual': ['r1 (Continuity)', 'r2 (x-momentum)', 'r3 (y-momentum)', 'r4 (z-momentum)'],
        'PINN Value': [
            f"{st.session_state.pinn_results['r1']:.6e}",
            f"{st.session_state.pinn_results['r2']:.6e}",
            f"{st.session_state.pinn_results['r3']:.6e}",
            f"{st.session_state.pinn_results['r4']:.6e}"
        ],
        'Ground Truth Value': [
            f"{st.session_state.gt_results['r1']:.6e}",
            f"{st.session_state.gt_results['r2']:.6e}",
            f"{st.session_state.gt_results['r3']:.6e}",
            f"{st.session_state.gt_results['r4']:.6e}"
        ]
    }
    
    df_residuals = pd.DataFrame(residual_data)
    st.dataframe(df_residuals, use_container_width=True, hide_index=True)
    
    # Enhanced plots
    st.subheader("📈 Visualization Plots")
    
    # Comparison charts
    fig_comparison = plot_comparison_charts(
        st.session_state.pinn_results,
        st.session_state.gt_results
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Residual statistics section
    st.subheader("📊 Residual Statistics")
    
    if st.button("📈 Compute Residual Statistics"):
        with st.spinner("Computing residual statistics on sample points..."):
            try:
                stats = compute_residual_statistics(
                    st.session_state.data['points'],
                    st.session_state.model,
                    st.session_state.normalizer,
                    st.session_state.nu,
                    st.session_state.rho,
                    device,
                    n_samples=1000
                )
                
                st.session_state.residual_stats = stats
                
                # Display statistics
                fig_stats = plot_residual_statistics(stats)
                st.plotly_chart(fig_stats, use_container_width=True)
                
                # Histograms
                fig_hist = plot_residual_histograms(stats)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error computing statistics: {str(e)}")
    
    if 'residual_stats' in st.session_state:
        stats = st.session_state.residual_stats
        
        # Display statistics table
        stats_rows = []
        for residual_name in ['r1', 'r2', 'r3', 'r4']:
            stats_rows.append({
                'Residual': residual_name,
                'Mean': f"{stats[residual_name]['mean']:.6e}",
                'Std Dev': f"{stats[residual_name]['std']:.6e}",
                'Max |Value|': f"{stats[residual_name]['max']:.6e}",
                'Min Value': f"{stats[residual_name]['min']:.6e}",
                'L2 Norm': f"{stats[residual_name]['l2_norm']:.6e}"
            })
        
        df_stats = pd.DataFrame(stats_rows)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Re-display plots
        fig_stats = plot_residual_statistics(stats)
        st.plotly_chart(fig_stats, use_container_width=True)
        
        fig_hist = plot_residual_histograms(stats)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Verification Report
    st.subheader("📄 Verification Report")
    
    if st.button("📥 Generate Report"):
        with st.spinner("Generating verification report..."):
            try:
                report = generate_verification_report(
                    st.session_state.pinn_results,
                    st.session_state.gt_results,
                    st.session_state.evaluated_coords,
                    st.session_state.residual_stats if 'residual_stats' in st.session_state else None,
                    st.session_state.nu,
                    st.session_state.rho
                )
                
                st.session_state.report = report
                
                # Display report
                st.markdown(report)
                
                # Download button
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name="pinn_verification_report.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
    
    if 'report' in st.session_state:
        st.markdown(st.session_state.report)
        st.download_button(
            label="💾 Download Report",
            data=st.session_state.report,
            file_name="pinn_verification_report.md",
            mime="text/markdown"
        )


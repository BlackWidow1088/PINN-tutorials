# PINN Model Verification - Streamlit App

Interactive web application for verifying Physics-Informed Neural Network (PINN) predictions against OpenFOAM simulation data for Navier-Stokes equations in a cylindrical pipe flow.

## Features

- **Interactive Coordinate Input**: Enter x, y, z coordinates to evaluate the PINN model
- **3D Visualization**: Visualize the pipe geometry with marked evaluation points
- **Comparison Analysis**: Compare PINN predictions vs ground truth values
- **Physics Residual Verification**: Check Navier-Stokes equation residuals
- **Statistical Analysis**: Compute residual statistics across sample points
- **Verification Reports**: Generate comprehensive verification reports with pass/fail criteria

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Local Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PINN-tutorials/cylinderCase
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Load Model**: Click "🔄 Load Model" in the sidebar to load the trained PINN model
2. **Load Data**: Click "📊 Load Simulation Data" to load OpenFOAM simulation data for the selected time step
3. **Enter Coordinates**: Input x, y, z coordinates in the main panel
4. **Evaluate**: Click "🔍 Evaluate" to get PINN predictions and compare with ground truth
5. **View Results**: 
   - See 3D pipe visualization with your point marked
   - Review comparison tables
   - Check Navier-Stokes residuals
   - Generate verification reports

## Project Structure

```
cylinderCase/
├── app.py                 # Main Streamlit application
├── pinn.py                # PINN model definition and training
├── verify_pinn.py         # Verification utilities
├── plotting_utils.py      # Enhanced plotting functions
├── report_generator.py   # Verification report generation
├── requirements.txt       # Python dependencies
├── pinn_model.pt         # Trained PINN model checkpoint
├── constant/             # OpenFOAM constant files
├── system/               # OpenFOAM system files
└── [time directories]/   # OpenFOAM time step data (0/, 100/, 200/, etc.)
```

## Dependencies

- `streamlit>=1.28.0` - Web application framework
- `torch>=2.0.0` - PyTorch for neural networks
- `numpy>=1.24.0` - Numerical computing
- `matplotlib>=3.7.0` - Plotting
- `plotly>=5.17.0` - Interactive 3D visualizations
- `pandas>=2.0.0` - Data manipulation
- `scipy>=1.10.0` - Scientific computing

## Deployment on Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Steps

1. **Push to GitHub**:
   - Create a new repository on GitHub
   - Push your code to the repository

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to: `cylinderCase/app.py`
   - Click "Deploy"

3. **Wait for Deployment**:
   - Streamlit Cloud will automatically install dependencies from `requirements.txt`
   - The app will be available at `https://<your-username>-<app-name>.streamlit.app`

### Important Notes for Streamlit Cloud

- Ensure `requirements.txt` includes all dependencies
- The repository size (~136MB) is acceptable for Streamlit Cloud
- First deployment may take a few minutes due to dependency installation
- Model file and data directories are included in the repository

## Model Information

- **Architecture**: 3 input → [50, 50, 50, 50] hidden layers → 4 output
- **Input**: Spatial coordinates (x, y, z)
- **Output**: Velocity components (u, v, w) and pressure (p)
- **Physics**: Incompressible Navier-Stokes equations
- **Training**: Physics-informed loss with data and physics constraints

## Verification Criteria

The app evaluates model performance using:

- **Prediction Accuracy**: Relative error < 5% for velocity and pressure
- **Physics Residuals**: |Residual| < 1e-3 for all Navier-Stokes equations
- **Statistical Analysis**: Mean and max residual statistics across sample points

## License

[Add your license here]

## Contact

[Add your contact information here]


# importing libraries
import sys
import os
import time
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# appending the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.pinn_model import build_pinn_model


# caching resources for the application
@st.cache_resource

# function for loading and compiling the pretrained Physics-Informed Neural Network (PINN)
def load_trained_model():
    # selecting execution hardware - CUDA GPU preferred if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiating the network topology
    model = build_pinn_model(layers=[2, 50, 50, 50, 50, 1]).to(device)

    # specifying the model path & loading weights into the model state dictionary 
    model_path = "models/pinn_trained.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@st.cache_data

# function for running a 1D Finite Difference Method (FDM) forward solver to get the baseline dataset.
def generate_cached_fdm():
    M, epsilon, Nx, Nt = 2.0, 0.01, 256, 4000
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    dx, dt = x[1] - x[0], t[1] - t[0]
    
    Sw = np.zeros((Nt, Nx))
    Sw[:, 0] = 1.0  
    
    for i in range(0, Nt - 1):
        for j in range(1, Nx - 1):
            sw_curr, sw_prev, sw_next = Sw[i, j], Sw[i, j-1], Sw[i, j+1]
            f_curr = (sw_curr**2) / (sw_curr**2 + M * (1 - sw_curr)**2)
            f_prev = (sw_prev**2) / (sw_prev**2 + M * (1 - sw_prev)**2)

            # first-order upwind difference approximation for the advection term (df/dx)
            df_dx = (f_curr - f_prev) / dx
            # second-order central difference approximation for the capillary diffusion term (d^2 Sw / dx^2)
            d2Sw_dx2 = (sw_next - 2 * sw_curr + sw_prev) / (dx**2)

            # explicit euler update step to project the saturation profile forward into time step i+1
            Sw[i+1, j] = sw_curr - dt * df_dx + epsilon * dt * d2Sw_dx2
        
        # applying a zero-gradient (Neumann) right boundary condition at the outlet (x = 1)
        Sw[i+1, Nx-1] = Sw[i+1, Nx-2]
        
    return x, t, Sw

model, device = load_trained_model()
x_fdm, t_fdm, Sw_fdm = generate_cached_fdm()

# setting up UI Layout & Tabs
st.set_page_config(page_title="Porous Media PINN Simulator", layout="centered", page_icon="🌊")
st.title("🌊Multiphase Flow PINN Simulator")

# adding an introductory text
st.markdown("""
A PINN web app predicting two-phase porous media flow. It solves the Buckley-Leverett equation to capture fluid shock fronts in real-time without labeled data.
""")

# setting up the tabs
tab1, tab2 = st.tabs(["Interactive Simulation", "Mathematical Framework"])

# tab 1: for the model prediction
with tab1:
    # add tab intro text
    st.markdown("Select a simulation time and click **Generate Profile** to view the wetting-phase saturation.")

    # setting up simulation time slider
    t_input = st.slider("Simulation Time (t)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)
    
    # setting the button placeholder
    btn_placeholder = st.empty()
    clicked = btn_placeholder.button("Generate Profile", key="btn_start")
    
    if clicked:
        btn_placeholder.button("Generating...", disabled=True, key="btn_generating")

        time.sleep(0.5)

        # locating the closest discrete timestamp index inside the FDM grid matching slider input
        idx_t = (np.abs(t_fdm - t_input)).argmin()
        actual_t = t_fdm[idx_t]

        # transforming the numpy array spatial grid into a column tensor
        x_tensor = torch.tensor(x_fdm[:, None], dtype=torch.float32).to(device)
        # creating tensor with the targeted time snapshot coordinate
        t_tensor = torch.full_like(x_tensor, actual_t)
        # forming the input structure [x, t]
        inputs = torch.cat([x_tensor, t_tensor], dim=1)

        # disabling gradient calculations to save memory & maximize speed
        with torch.no_grad():
            Sw_pred = model(inputs).cpu().numpy().flatten()

        # isolating ground truth time array from the numerical solver 
        Sw_true = Sw_fdm[idx_t, :]

        # calculating the relative L2 norm error
        step_l2 = np.linalg.norm(Sw_true - Sw_pred) / np.linalg.norm(Sw_true)
        
        # displaying L2 norm error
        st.metric(label="Relative L2 Error", value=f"{step_l2*100:.2f}%")
        
        # setting up Plotly Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_fdm, y=Sw_true, mode='lines', name='FDM (Ground Truth)', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=x_fdm, y=Sw_pred, mode='lines', name='PINN Prediction', line=dict(color='red', width=3, dash='dash')))

        fig.update_layout(
                title=f"Shock Profile at t={actual_t:.2f}",
                xaxis_title="Spatial Distance (x)",
                yaxis_title="Wetting Phase Saturation (Sw)",
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True), 
                template="plotly_white",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
        st.plotly_chart(fig, use_container_width=True)
        
        # compiling results into a structured Pandas DataFrame for optional download
        df_export = pd.DataFrame({
            "Spatial_Distance_x": x_fdm,
            "FDM_Ground_Truth_Sw": Sw_true,
            "PINN_Prediction_Sw": Sw_pred
        })
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name=f"shock_profile_t{actual_t:.2f}.csv",
            mime="text/csv"
        )

        # displaying success message
        st.toast("Profile successfully generated!", icon="✅", duration=2)
        
        btn_placeholder.button("Generate Profile", key="btn_done")

# tab 2: for the mathematical framework
with tab2:
    st.subheader("Governing Physics")
    st.markdown("The network predicts the spatial-temporal distribution of $S_w(x,t)$ by minimizing the residual of the Buckley-Leverett equation with capillary diffusion:")
    st.latex(r"\frac{\partial S_w}{\partial t} + \frac{\partial f(S_w)}{\partial x} - \epsilon \frac{\partial^2 S_w}{\partial x^2} = 0")
    
    st.subheader("Architecture")
    st.markdown("""
    * **Methodology:** Physics-Informed Neural Network (PINN) via `torch.autograd`
    * **Network Capacity:** 5 Hidden Layers $\\times$ 50 Neurons
    * **Activation Function:** SiLU (Swish) to mitigate spectral bias at the shock front
    * **Optimization:** Adam followed by L-BFGS with targeted collocation sampling
    * **Overall Validation:** Global relative $L_2$ error of $< 3.0\\%$ across the continuous $(x,t)$ domain.
    """)
# importing libraries
import torch
import torch.nn as nn

# building the PINN model
def build_pinn_model(layers=[2, 50, 50, 50, 50, 1]):
    """
    Builds the PINN using SiLU activations.
    """
    module_list = []
    for i in range(len(layers) - 1):
        linear_layer = nn.Linear(layers[i], layers[i+1])
        nn.init.xavier_normal_(linear_layer.weight)
        nn.init.zeros_(linear_layer.bias)
        
        module_list.append(linear_layer)
        
        # Apply SiLU (Swish) instead of Tanh to hidden layers
        if i < len(layers) - 2:
            module_list.append(nn.SiLU())
            
    return nn.Sequential(*module_list)

# function to calculate the residual loss
def compute_physics_loss(model, x, t, M=2.0, epsilon=0.01):
    """
    Computes the Buckley-Leverett PDE residual loss computationally using Automatic 
    Differentiation (autograd).
    """
    # enabling gradient tracking for the spatial and temporal input coordinates
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    # concatenating spatial and temporal inputs with the feature dimension 
    inputs = torch.cat([x, t], dim=1)
    
    # predicting the Wetting-Phase Saturation (Sw_hat)
    Sw_hat = model(inputs)
    
    # deriving the temporal derivative: d(Sw_hat)/dt
    dSw_dt = torch.autograd.grad(
        outputs=Sw_hat, 
        inputs=t, 
        grad_outputs=torch.ones_like(Sw_hat), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # calculating the nonlinear fractional flow function f(Sw_hat)
    f_Sw = (Sw_hat**2) / (Sw_hat**2 + M * (1 - Sw_hat)**2)
    
    # deriving the spatial derivative of fractional flow (advection term): d(f(Sw_hat))/dx
    df_dx = torch.autograd.grad(
        outputs=f_Sw, 
        inputs=x, 
        grad_outputs=torch.ones_like(f_Sw), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # deriving the first spatial derivative of Sw: d(Sw_hat)/dx
    dSw_dx = torch.autograd.grad(
        outputs=Sw_hat, 
        inputs=x, 
        grad_outputs=torch.ones_like(Sw_hat), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # deriving the second spatial derivative of Sw (capillary diffusion term): d2(Sw_hat)/dx2
    d2Sw_dx2 = torch.autograd.grad(
        outputs=dSw_dx, 
        inputs=x, 
        grad_outputs=torch.ones_like(dSw_dx), 
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # formulating the complete Buckley-Leverett PDE Residual Equation
    pde_residual = dSw_dt + df_dx - epsilon * d2Sw_dx2
    
    # returning the Mean Squared Error (MSE) of the residual
    return torch.megan(pde_residual**2)
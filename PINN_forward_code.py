#%% Import 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import csv
import sys
import math
import glob
import time
import psutil
import shutil
import random
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import json


from sklearn.metrics import mean_squared_error, r2_score
from scipy.ndimage import gaussian_filter


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




#%% Preparation



head_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(head_path)



f  = np.load("original_f.npy")
v  = np.load("original_v.npy")
idx_node_top    = np.load("idx_node_top.npy")
idx_node_bot    = np.load("idx_node_bot.npy")
idx_node_right  = np.load("idx_node_right.npy")
idx_node_left   = np.load("idx_node_left.npy")

print(f"f.shape              : {f.shape}")
print(f"v.shape              : {v.shape}")
print(f"idx_node_top.shape   : {idx_node_top.shape}")
print(f"idx_node_bot.shape   : {idx_node_bot.shape}")
print(f"idx_node_right.shape : {idx_node_right.shape}")
print(f"idx_node_left.shape  : {idx_node_left.shape}")


print("-"*50)




# Set seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1)


#%%


v = torch.tensor(v, dtype=torch.float32, requires_grad=True, device=device)
f = torch.tensor(f, dtype=torch.int64, requires_grad=False, device=device) 

idx_node_top   = torch.tensor(idx_node_top,   dtype=torch.int64, requires_grad=False, device=device)
idx_node_bot   = torch.tensor(idx_node_bot,   dtype=torch.int64, requires_grad=False, device=device)
idx_node_right = torch.tensor(idx_node_right, dtype=torch.int64, requires_grad=False, device=device)
idx_node_left  = torch.tensor(idx_node_left,  dtype=torch.int64, requires_grad=False, device=device)


f_node_coord=v[f]


#%%

def C0_T3_shape_function(element_coords, L2, L3):

    n_elem = element_coords.shape[0]

    x1 = element_coords[:, 0, 0]
    y1 = element_coords[:, 0, 1]
    x2 = element_coords[:, 1, 0]
    y2 = element_coords[:, 1, 1]
    x3 = element_coords[:, 2, 0]
    y3 = element_coords[:, 2, 1]

    AREA = 0.5*( x2*y3 - x3*y2 + x3*y1 - x1*y3 + x1*y2 - x2*y1 )

    L1 = 1.0 - L2 - L3  # (n,)

    # shape function: N1=L1, N2=L2, N3=L3
    N1 = np.ones(n_elem) * L1
    N2 = np.ones(n_elem) * L2
    N3 = np.ones(n_elem) * L3
    N_all = np.stack([N1, N2, N3], axis=1)  # (n,3)

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    twoA = 2.0 * AREA  # (n,)
    inv_twoA = 1 / twoA

    dN1_dx = b1 * inv_twoA
    dN2_dx = b2 * inv_twoA
    dN3_dx = b3 * inv_twoA

    dN1_dy = c1 * inv_twoA
    dN2_dy = c2 * inv_twoA
    dN3_dy = c3 * inv_twoA

    dNdx_all = np.stack([dN1_dx, dN2_dx, dN3_dx], axis=1)  # (n,3)
    dNdy_all = np.stack([dN1_dy, dN2_dy, dN3_dy], axis=1)  # (n,3)

    jacobian_all = twoA  # (n,)

    return N_all, dNdx_all, dNdy_all, jacobian_all


def GQ_1():
    """
    Dunavant degree=2, 3-point rule.
    """
    data = np.array([
        [1/3, 1/3, 1/2],   # (x,y,w)
    ])
    points  = data[:, :2]
    weights = data[:, 2]
    return points, weights


def GQ_3():
    """
    Dunavant degree=2, 3-point rule.
    """
    data = np.array([
        [1/6, 1/6, 1/6],   # (x,y,w)
        [4/6, 1/6, 1/6],
        [1/6, 4/6, 1/6],
    ])
    points  = data[:, :2]
    weights = data[:, 2]
    return points, weights



#%%

f_node_coord = v[f].detach().cpu().numpy()

gaussian_points, gaussian_weights = GQ_1()

areas_divided = []
B_n_ = []
B_u_ = []
B_e_ = []

for (L2,L3), weight in zip(gaussian_points, gaussian_weights):
    
    N, dNdx, dNdy, jacobian = C0_T3_shape_function(f_node_coord, L2, L3)
    
    areas_divided.append(jacobian*weight)
    
    n_el = f_node_coord.shape[0]
    
    offset_B_u = 3
    B_u = np.zeros((n_el, 3, 2 * offset_B_u))
    B_u[:, 0, 0*offset_B_u:1*offset_B_u] = dNdx 
    B_u[:, 1, 1*offset_B_u:2*offset_B_u] = dNdy
    B_u[:, 2, 0*offset_B_u:1*offset_B_u] = dNdy
    B_u[:, 2, 1*offset_B_u:2*offset_B_u] = dNdx 
    
    
    offset_B_n = 3
    B_n = np.zeros((n_el, 6, 3 * offset_B_n))
    B_n[:, 0, 0*offset_B_n:1*offset_B_n] = dNdx 
    B_n[:, 1, 1*offset_B_n:2*offset_B_n] = dNdx
    B_n[:, 2, 2*offset_B_n:3*offset_B_n] = dNdx
    B_n[:, 3, 0*offset_B_n:1*offset_B_n] = dNdy
    B_n[:, 4, 1*offset_B_n:2*offset_B_n] = dNdy 
    B_n[:, 5, 2*offset_B_n:3*offset_B_n] = dNdy


    offset_B_e = 3
    B_e = np.zeros((n_el, 2, 1 * offset_B_e))
    B_e[:, 0, 0*offset_B_e:1*offset_B_e] = - dNdx 
    B_e[:, 1, 0*offset_B_e:1*offset_B_e] = - dNdy

    B_u_.append(B_u)
    B_n_.append(B_n)
    B_e_.append(B_e)



areas_divided = np.stack(areas_divided)
B_u = np.stack(B_u_)
B_n = np.stack(B_n_)
B_e = np.stack(B_e_)



areas_divided  = torch.tensor(areas_divided,  dtype=torch.float32, requires_grad=False, device=device).contiguous().view(-1)
B_u   = torch.tensor(B_u_,   dtype=torch.float32, requires_grad=False, device=device).contiguous().view(-1, 3,6)
B_n   = torch.tensor(B_n_,   dtype=torch.float32, requires_grad=False, device=device).contiguous().view(-1, 6,9)
B_e = torch.tensor(B_e_, dtype=torch.float32, requires_grad=False, device=device).contiguous().view(-1, 2,3)

X_infer = v.clone()

print("areas_divided.shape", areas_divided.shape)
print("B_u.shape", B_u.shape)
print("B_n.shape", B_n.shape)
print("B_e.shape", B_e.shape)
print("X_infer.shape", X_infer.shape)



#%%


def compute_autograd_gradients(output_tensor, input_tensor):
    """
    Computes the Jacobian of output_tensor w.r.t input_tensor efficiently while ensuring identical results.
    Fully vectorized implementation while keeping gradients intact for training.
    """
    input_tensor.requires_grad_(True)

    batch_size, output_dim = output_tensor.shape  # (n, m)
    input_dim = input_tensor.shape[1]  # (n, d)

    # Flatten output tensor for batch-wise processing
    output_flat = output_tensor.view(batch_size, output_dim)  # (n, m)

    # Create grad_outputs as a list of one-hot vectors to compute gradients separately
    grad_outputs = torch.eye(output_dim, dtype=output_tensor.dtype, device=output_tensor.device)  # (m, m)

    # Compute the Jacobian by iterating over output dimensions (keeping gradients intact)
    jacobian_list = []
    for i in range(output_dim):
        grad = torch.autograd.grad(
            outputs=output_flat, 
            inputs=input_tensor, 
            grad_outputs=grad_outputs[i].expand(batch_size, output_dim),  # (n, m)
            create_graph=True, 
            retain_graph=True
        )[0]  # Shape: (n, d)

        jacobian_list.append(grad.unsqueeze(2))  # Shape: (n, d, 1)

    # Concatenate results to form full Jacobian matrix (n, d, m)
    jacobian = torch.cat(jacobian_list, dim=2)

    # Ensure correct reshape to (n, m, d)
    gradients = jacobian.permute(0, 2, 1).contiguous()  # Reshape from (n, d, m) to (n, m, d)

    return gradients


class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        # return F.tanh(x)
        # return F.softplus(x)
        # return F.mish(x)
        return x * torch.sigmoid(x)
        # return torch.sin(x)


class NN_1(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=6):
        super(NN_1, self).__init__()

        self.NN_displacement = self._make_subnet(hidden_dim, num_layers, output_dim=2)

        self.NN_V = self._make_subnet(hidden_dim, num_layers, output_dim=1)

        self.apply(self._init_weights)

    def _make_subnet(self, hidden_dim, num_layers, output_dim):
        layers = [nn.Linear(2, hidden_dim), CustomActivation()]
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), CustomActivation()])
        layers.append(nn.Linear(hidden_dim, output_dim)) 
        return nn.Sequential(*layers)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, X):
        x, y = X[:, 0], X[:, 1]
        
        x_norm = x / 5
        y_norm = y 
        
        X_input = torch.stack((x_norm , y_norm ), dim=1) 
        
        displacement_outputs = self.NN_displacement(  X_input )
        ux_out = displacement_outputs[:, 0] * x_norm
        uy_out = displacement_outputs[:, 1] * x_norm 

        V_out = self.NN_V(  X_input  ).squeeze() * y_norm * (1 - y_norm) + y_norm * 0.01

        return torch.stack([ux_out, uy_out, V_out], dim=1)


def plot_loss_history(loss_history):
    plt.figure(figsize=(8, 6), dpi=300)

    valid_losses = {} 

    for key, values in loss_history.items():
        if values and np.any(np.array(values) != 0):
            valid_losses[key] = values

    if not valid_losses:  
        print("⚠ No valid loss data to plot.")
        return

    for key, values in valid_losses.items():
        plt.plot(values, label=key, alpha=0.7)

    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Loss History")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    min_loss = min(min(values) for values in valid_losses.values())
    max_loss = max(max(values) for values in valid_losses.values())

    plt.ylim(min_loss*1e-1, max_loss * 10)  

    plt.show()
    plt.close()


def plot_nodal_value(x, y, value, title="Nodal Value Plot", cmap="RdBu_r", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    sc = ax.scatter(
        x.cpu().detach().numpy(), 
        y.cpu().detach().numpy(), 
        c=value.cpu().detach().numpy(), 
        cmap=cmap, 
        s=20, 
        # edgecolors="k", 
        alpha=0.8
    )
    
    cbar = plt.colorbar(sc, ax=ax)  
    cbar.set_label("Value", fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    if ax is None:
        plt.tight_layout()
        plt.show()


#%%


# =========================================================================
#  Material property
# =========================================================================
E  = torch.tensor(139e9,  dtype=torch.float32, requires_grad=False, device=device)
nu = torch.tensor(0.3,  dtype=torch.float32, requires_grad=False, device=device)

sl     = torch.tensor(400e-9,  dtype=torch.float32, requires_grad=False, device=device)
cond_k = torch.tensor(1e-9,  dtype=torch.float32, requires_grad=False, device=device)
f1     = torch.tensor(1e-6,  dtype=torch.float32, requires_grad=False, device=device) * 1
f2     = torch.tensor(1e-6,  dtype=torch.float32, requires_grad=False, device=device) * 1


L_ref = 1000e-9
E_ref = E 
k_ref = cond_k 
flexo_ref = L_ref * (E_ref * k_ref) ** 0.5
vol_ref = L_ref * (E_ref / k_ref) ** 0.5

E = E / E_ref
cond_k = cond_k / k_ref
sl = sl / L_ref
f1 = f1 / flexo_ref
f2 = f2 / flexo_ref


lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu  = E / (2 * (1 + nu))

C_uu = torch.tensor([
    [lam + 2 * mu, lam,          0   ],
    [lam,          lam + 2 * mu,  0   ],
    [0,            0,             mu ]
], dtype=torch.float32, device=device)  # Plane strain condition

C_nn = torch.tensor([
    [lam + 2 * mu,  lam,          0,           0,           0,          0   ],
    [lam,          lam + 2 * mu,  0,           0,           0,          0   ],
    [0,            0,            mu,          0,           0,          0   ],
    [0,            0,            0,           lam + 2 * mu, lam,        0   ],
    [0,            0,            0,           lam,         lam + 2 * mu, 0  ],
    [0,            0,            0,           0,           0,          mu  ]
], dtype=torch.float32, device=device) * sl * sl


C_ee = torch.tensor([
    [cond_k, 0     ],
    [0,      cond_k]
], dtype=torch.float32, device=device)

C_en = torch.tensor([
    [f1 + 2 * f2,  f1,   0,   0,    0,    f2 ],
    [0,           0,    f2,   f1,  f1 + 2 * f2,  0 ]
], dtype=torch.float32, device=device)



#%%

def closure():
    
    # =============================================================================
    # Internal 
    # =============================================================================
    X = X_infer.clone()  # (num_elements * 3, 2)
    Y = model_1(X)  # (num_elements * 3, 3)
    dY_dX = compute_autograd_gradients(Y, X)  # (num_elements * 3, 3, 2)
    
    ux, uy, V = Y[:, 0], Y[:, 1], Y[:, 2]
    duxdx, duydx, dVdx = dY_dX[:, 0, 0], dY_dX[:, 1, 0], dY_dX[:, 2, 0]
    duxdy, duydy, dVdy = dY_dX[:, 0, 1], dY_dX[:, 1, 1], dY_dX[:, 2, 1]
    
    ###### For output  --------------------------------
    disp_sol = torch.stack((ux, uy), dim=1)
    strain_sol = torch.stack((duxdx, duydy, duxdy + duydx), dim=1)
    elect_field_sol = torch.stack((-dVdx, -dVdy), dim=1)
    ddu = compute_autograd_gradients(strain_sol, X)
    strain_grad_sol = torch.cat((ddu[:, :, 0], ddu[:, :, 1]), dim=1)
    stress = torch.matmul(strain_sol, C_uu.T)
    tau = torch.matmul(strain_grad_sol, C_nn.T) - torch.matmul(elect_field_sol, C_en)
    elect_disp = torch.matmul(elect_field_sol, C_ee.T) + torch.matmul(strain_grad_sol, C_en.T)
    ###### ---------------------------------------------
    
    
    disp_sol = torch.stack((ux, uy), dim=1)
    strain_sol = torch.stack((duxdx, duydy, duxdy + duydx), dim=1)
    
    disp_el   = torch.cat((ux[f], uy[f]), dim=1)
    strain_el = torch.cat((duxdx[f], duydy[f], duxdy[f] + duydx[f]), dim=1)
    V_el      = V[f]
    
    
    strain_numerical      = torch.einsum('eij, ej->ei', B_u, disp_el)
    strain_grad_numerical = torch.einsum('eij, ej->ei', B_n, strain_el)
    elect_field_numerical = torch.einsum('eij, ej->ei', B_e, V_el)
    
    mech_int_energy_1 = torch.einsum('ei, ij, ej, e->', strain_numerical, C_uu, strain_numerical, areas_divided) / 2
    mech_int_energy_2 = torch.einsum('ei, ij, ej, e->', strain_grad_numerical, C_nn, strain_grad_numerical, areas_divided) / 2
    elect_int_energy  = torch.einsum('ei, ij, ej, e->', elect_field_numerical, C_ee, elect_field_numerical, areas_divided) / 2
    couple_energy     = torch.einsum('ei, ij, ej, e->', elect_field_numerical, C_en, strain_grad_numerical, areas_divided) 
    
    internal_energy = mech_int_energy_1 + mech_int_energy_2 - elect_int_energy - couple_energy
    
    # aux = torch.mean( (strain_sol[f].mean(dim=1)-strain_numerical) ** 2 )
    
    # internal_energy = internal_energy + aux
    
    # =============================================================================
    # External 
    # =============================================================================
    mech_ext_energy_1 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    mech_ext_energy_2 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    elect_ext_energy = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
    
    external_energy = mech_ext_energy_1 + mech_ext_energy_2 - elect_ext_energy

    loss = internal_energy - external_energy

    return (( loss ) ,
            ( mech_int_energy_1, mech_int_energy_2, elect_int_energy, couple_energy ,
             mech_ext_energy_1, mech_ext_energy_2, elect_ext_energy ),
            ( X, disp_sol, V, strain_sol, strain_grad_sol, elect_field_sol, stress, tau, elect_disp ), 
            )


#%% EMA

model_1 = NN_1().to(device)  # output [x-dir displacement, y-dir displacement, Voltage]
model_1.train()


loss_history = {
    "final_loss": [],
    "mech_int_energy_1": [],
    "mech_int_energy_2": [],
    "elect_int_energy": [],
    "couple_energy": [],
    "mech_ext_energy_1": [],
    "mech_ext_energy_2": [],
    "elect_ext_energy": [],
    "grad_norm_disp": [],
    "grad_norm_V": [],
}



optimizer_1 = torch.optim.Adam(model_1.NN_displacement.parameters(), lr=1e-3)
optimizer_2 = torch.optim.Adam(model_1.NN_V.parameters(), lr=1e-3)


start_time = time.time()
num_epochs_adam = 100000

for epoch in range(1, num_epochs_adam + 1):

    closure_output = closure()
    loss = closure_output[0]
    mech_int_energy_1, mech_int_energy_2, elect_int_energy, couple_energy ,mech_ext_energy_1, mech_ext_energy_2, elect_ext_energy = closure_output[1]
    X, disp_sol, V, strain_sol, strain_grad_sol, elect_field_sol, stress, tau, elect_disp = closure_output[2]
        
    disp_loss = loss *1e5
    V_loss = -loss   *1e5

    
    
    theta_t = {name: param.clone().detach() for name, param in model_1.NN_displacement.named_parameters()}
    phi_t = {name: param.clone().detach() for name, param in model_1.NN_V.named_parameters()}

    # displacement
    optimizer_1.zero_grad()
    grads_disp = torch.autograd.grad(disp_loss , model_1.NN_displacement.parameters(), create_graph=True, allow_unused=True)
    for param, grad in zip(model_1.NN_displacement.parameters(), grads_disp):
        if grad is not None:
            param.grad = grad.detach()
    optimizer_1.step()

    # voltage
    optimizer_2.zero_grad()
    grads_v = torch.autograd.grad(V_loss , model_1.NN_V.parameters(), create_graph=True, allow_unused=True)
    for param, grad in zip(model_1.NN_V.parameters(), grads_v):
        if grad is not None:
            param.grad = grad.detach()
    optimizer_2.step()
    

    beta_ema = 0.95 if epoch > 1 else 0 
    with torch.no_grad():
        for name, param in model_1.NN_displacement.named_parameters():
            param.copy_(beta_ema * theta_t[name] + (1 - beta_ema) * param)  # EMA 

        for name, param in model_1.NN_V.named_parameters():
            param.copy_(beta_ema * phi_t[name] + (1 - beta_ema) * param)  # EMA 

    loss_history["final_loss"].append( loss.item()  )
    loss_history["mech_int_energy_1"].append(mech_int_energy_1.item())
    loss_history["mech_int_energy_2"].append(mech_int_energy_2.item())
    loss_history["elect_int_energy"].append(elect_int_energy.item())
    loss_history["couple_energy"].append(couple_energy.item())
    loss_history["mech_ext_energy_1"].append(mech_ext_energy_1.item())
    loss_history["mech_ext_energy_2"].append(mech_ext_energy_2.item())
    loss_history["elect_ext_energy"].append(elect_ext_energy.item())


    if epoch % 100 == 0:
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"Adam solver : Epoch {epoch} \t total_loss = {loss:.4e} \t Elapsed Time = {elapsed_time_str}")

        
    if epoch % 1000 == 0:
        plot_loss_history(loss_history)
        
                
        
    if epoch % 1000 == 0:

        
        fig, ax = plt.subplots(1, 3, figsize=(24, 2), dpi=300)
        plot_nodal_value(X[:,0]+disp_sol[:,0], X[:,1]+disp_sol[:,1], disp_sol[:,0], title="X displacement", cmap="RdBu_r", ax=ax[0])
        plot_nodal_value(X[:,0]+disp_sol[:,0], X[:,1]+disp_sol[:,1], disp_sol[:,1], title="Y displacement", cmap="RdBu_r", ax=ax[1])
        plot_nodal_value(X[:,0]+disp_sol[:,0], X[:,1]+disp_sol[:,1], V            , title="Voltage", cmap="RdBu_r", ax=ax[2])
        plt.show()
        plt.close()
        
        
        fig, ax = plt.subplots(4,3, figsize=(32, 8), dpi=300)
        plot_nodal_value(X[:,0], X[:,1], strain_sol[:,0], title="strain xx", cmap="RdBu_r", ax=ax[0,0])
        plot_nodal_value(X[:,0], X[:,1], strain_sol[:,1], title="strain yy", cmap="RdBu_r", ax=ax[0,1])
        plot_nodal_value(X[:,0], X[:,1], strain_sol[:,2], title="strian xy", cmap="RdBu_r", ax=ax[0,2])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,0], title="strain_grad xx_x", cmap="RdBu_r", ax=ax[1,0])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,1], title="strain_grad yy_x", cmap="RdBu_r", ax=ax[1,1])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,2], title="strain_grad xy_x", cmap="RdBu_r", ax=ax[1,2])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,3], title="strain_grad xx_y", cmap="RdBu_r", ax=ax[2,0])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,4], title="strain_grad yy_y", cmap="RdBu_r", ax=ax[2,1])
        plot_nodal_value(X[:,0], X[:,1], strain_grad_sol[:,5], title="strain_grad xy_y", cmap="RdBu_r", ax=ax[2,2])    
        plot_nodal_value(X[:,0], X[:,1], elect_field_sol[:,0], title="X electric field", cmap="RdBu_r", ax=ax[3,0])
        plot_nodal_value(X[:,0], X[:,1], elect_field_sol[:,1], title="Y electric field", cmap="RdBu_r", ax=ax[3,1])
        plt.show()
        plt.close()
        
        fig, ax = plt.subplots(4,3, figsize=(32, 8), dpi=300)
        plot_nodal_value(X[:,0], X[:,1], stress[:,0], title="stress xx", cmap="RdBu_r", ax=ax[0,0])
        plot_nodal_value(X[:,0], X[:,1], stress[:,1], title="stress yy", cmap="RdBu_r", ax=ax[0,1])
        plot_nodal_value(X[:,0], X[:,1], stress[:,2], title="stress xy", cmap="RdBu_r", ax=ax[0,2])
        plot_nodal_value(X[:,0], X[:,1], tau[:,0], title="tau xx_x", cmap="RdBu_r", ax=ax[1,0])
        plot_nodal_value(X[:,0], X[:,1], tau[:,1], title="tau yy_x", cmap="RdBu_r", ax=ax[1,1])
        plot_nodal_value(X[:,0], X[:,1], tau[:,2], title="tau xy_x", cmap="RdBu_r", ax=ax[1,2])
        plot_nodal_value(X[:,0], X[:,1], tau[:,3], title="tau xx_y", cmap="RdBu_r", ax=ax[2,0])
        plot_nodal_value(X[:,0], X[:,1], tau[:,4], title="tau yy_y", cmap="RdBu_r", ax=ax[2,1])
        plot_nodal_value(X[:,0], X[:,1], tau[:,5], title="tau xy_y", cmap="RdBu_r", ax=ax[2,2])    
        plt.show()
        plt.close()
        






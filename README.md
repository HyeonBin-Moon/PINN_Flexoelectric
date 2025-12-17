# Energy-Based PINN for Flexoelectric Problems

This repository provides the official implementation of the paper:

**Hyeonbin Moon†, Donggeun Park†, Jinwook Yeo, Seunghwa Ryu\***
*An Energy-Based Physics-Informed Neural Network Framework for Solving Forward and Inverse Flexoelectric Problems*,  
International Journal for Numerical Methods in Engineering (under revision).

A preprint version of the paper is available at:  
https://doi.org/10.48550/arXiv.2506.21810

This code implements an energy-based Physics-Informed Neural Network (PINN) framework
for solving both **forward** and **inverse** problems in linear flexoelectricity.

---

## Overview

Flexoelectricity involves the coupling between strain gradients and electric polarization,
leading to governing equations with fourth-order spatial derivatives.
Such high-order partial differential equations pose significant challenges for conventional
PINN formulations, particularly in terms of numerical stability and boundary condition enforcement.

To address these issues, the proposed framework adopts a **deep energy method (DEM)**,
in which the governing equations are reformulated through the total potential energy
and solved as a **saddle-point optimization problem**.

- **Forward problem**:  
  Predict displacement and electric potential fields from prescribed mechanical
  and electrical boundary conditions.

- **Inverse problem**:  
  Identify unknown flexoelectric coefficients from sparse electric potential measurements.

Finite element–based numerical quadrature is employed for stable and consistent
evaluation of the energy functional, and **hard constraints** are used to enforce
Dirichlet boundary conditions exactly.

---

## Scope of the Provided Examples

This repository provides representative example codes corresponding to specific
numerical configurations investigated in the paper.

- **Forward problem**:  
  The forward example included in this repository corresponds to the  
  **converse flexoelectric effect**, in which an externally applied electric potential
  induces mechanical deformation through electromechanical coupling.
  This setup is consistent with the numerical example presented in the paper
  for the converse flexoelectric response.

- **Inverse problem**:  
  The inverse example focuses on the identification of flexoelectric coefficients
  in the case where **both coefficients are positive-valued**
  (i.e., \( f_1 > 0 \) and \( f_2 > 0 \)).
  This configuration corresponds to one of the representative inverse test cases
  reported in the paper and demonstrates stable and accurate parameter recovery
  under well-posed conditions.

Additional forward and inverse configurations discussed in the paper
(e.g., the direct flexoelectric effect or cases with mixed-sign coefficients)
can be implemented within the same framework with straightforward modifications.

---

## Repository Structure

## Repository Structure

```text
├── forward/
│   ├── PINN_forward_code.py      # Forward problem: converse flexoelectric effect
│   ├── original_v.npy            # Mesh node coordinates (vertices)
│   ├── original_f.npy            # Mesh connectivity (elements)
│   ├── idx_node_left.npy         # Boundary node indices: left edge
│   ├── idx_node_right.npy        # Boundary node indices: right edge
│   ├── idx_node_top.npy          # Boundary node indices: top edge
│   └── idx_node_bot.npy          # Boundary node indices: bottom edge
├── inverse/
│   ├── PINN_inverse_code.py      # Inverse problem: coefficient identification
│   ├── original_v.npy            # Mesh node coordinates (vertices)
│   ├── original_f.npy            # Mesh connectivity (elements)
│   ├── idx_node_left.npy         # Boundary node indices: left edge
│   ├── idx_node_right.npy        # Boundary node indices: right edge
│   ├── idx_node_top.npy          # Boundary node indices: top edge
│   ├── idx_node_bot.npy          # Boundary node indices: bottom edge
│   └── top_voltage_data.npz      # Electric potential measurements on the top boundary
├── requirements.txt
└── README.md

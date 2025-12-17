# Energy-Based PINN for Forward and Inverse Flexoelectric Problems

This repository provides the official implementation of the paper:


**H. Moon†, D. Park†, J. Yeo, S. Ryu**,  
*An Energy-Based Physics-Informed Neural Network Framework for Solving Forward and Inverse Flexoelectric Problems*,  
International Journal for Numerical Methods in Engineering (under revision).

This code implements an energy-based Physics-Informed Neural Network (PINN) framework
for solving both **forward** and **inverse** problems in linear flexoelectricity.

---

## Overview

Flexoelectric problems involve fourth-order partial differential equations due to strain-gradient effects,
which pose significant challenges for conventional PINN formulations.
To address this, the proposed framework adopts a **deep energy method (DEM)** based on
the total potential energy, formulated as a **saddle-point problem**.

- **Forward problem**:  
  Predict displacement and electric potential fields from prescribed boundary conditions.
- **Inverse problem**:  
  Identify unknown flexoelectric coefficients from sparse electric potential measurements.

Finite element–based numerical quadrature is employed for stable energy evaluation,
and hard constraints are used to enforce Dirichlet boundary conditions exactly.

---


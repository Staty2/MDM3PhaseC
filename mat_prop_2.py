#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:49:54 2025

@author: samprior
"""

import numpy as np
import matplotlib.pyplot as plt

# Material properties: E (Pa), v, rho (kg/m³)
materials = {
    "Bone": {"E": 1.5e10, "v": 0.22, "rho": 2000},
    "EVA": {"E": 5e5, "v": 0.3, "rho": 120},
}

# Skull properties
E_skull = 1.5e10  # Young's Modulus (Pa)
v_skull = 0.22  # Poisson's Ratio
rho_skull = 2000  # Density (kg/m³)

# Wave speed in skull
c_skull = np.sqrt(E_skull / rho_skull)   
Z_skull = rho_skull * c_skull   # Impedance of the skull

# Store results
transmission_results = {}

for material, props in materials.items():
    E, v, rho = props["E"], props["v"], props["rho"]

    # Wave speed of the material
    c_mat = np.sqrt(E / rho)   
    Z_mat = rho * c_mat  # Impedance of the material

    # Energy Transmission Coefficient
    T = (4 * Z_mat * Z_skull) / (Z_mat + Z_skull)**2
    
    # Store in dictionary
    transmission_results[material] = T
    
    # Debugging: Print values
    print(f"Material: {material}, Z_mat: {Z_mat:.2e}, Z_skull: {Z_skull:.2e}, T: {T:.4f}")

# Bar chart
plt.figure(figsize=(8, 5))
plt.bar(transmission_results.keys(), transmission_results.values(), color=plt.cm.viridis(np.linspace(0, 1, len(transmission_results))))
plt.ylabel("Energy Transmission Coefficient")
plt.title("Wave Transmission Through Different Materials")
plt.ylim(0, 1)  # Ensure values stay within valid range
plt.show()

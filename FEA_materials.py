import numpy as np
import matplotlib.pyplot as plt
# for 0.003s with an impactor v = 4ms^-1
# viscoelastic = PXD, SBT, VNF, MFM
# Define foam material types
materials = [
    "Ethylene Vinyl Acetate", "Sorbothane", "Flexible PU Foam", "Rigid PU Foam",
    "Memory Foam", "Poron XRD", "Vinyl Nitrile Foam"
]

# Thickness range (in mm)
thicknesses = [0.5, 1, 2, 3]

deformation_PXD = [0.21, 0.11, 0.046, 0.011]
deformation_EVA = [0.282137, 0.163418, 7.71E-02, 4.78E-02]
deformation_VNF = [0.35, 0.183, 0.085, 0.053]
deformation_FPU = [0.441368, 0.317077, 0.219824, 0.162786]
deformation_MFM = [0.552, 0.396, 0.275, 0.23]
deformation_SBT = [0.42, 0.28, 0.133, 0.09]
deformation_RPF = [0.617305, 0.526304, 0.45031, 0.422632]

stress_EVA = [196.68, 121.91, 64.06, 39.44]
stress_PXD = [147.51, 91.43, 38.05, 29.58 ]
stress_SBT = [109.32, 76.35, 32.38, 18.94]
stress_MFM = [312.29, 181.75, 112.72, 97.15]
stress_VNF = [163.901, 97.591, 47.3845, 35.8679 ]
stress_FPU = [293.157, 174.061, 121.399, 104.098 ]
stress_RPF = [412.304, 356.076, 253.216, 222.151]

acceleration_SBT = [120, 58, 42, 15]
acceleration_PXD = [270, 72, 58, 25]
acceleration_VNF = [250, 68, 50, 42]
acceleration_MFM = [260, 123, 82, 64]
acceleration_EVA = [292.728, 144.074, 91.8661, 81.734]
acceleration_FPU = [741.959, 398.674, 219.498, 179.702]
acceleration_RPF = [734.716, 804.449, 610.108, 462.346]



plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Deformation vs Thickness
plt.figure(figsize=(6, 5))
plt.plot(thicknesses, deformation_PXD, '-o',label="PXD")
plt.plot(thicknesses, deformation_EVA, '-o',label="EVA")
plt.plot(thicknesses, deformation_VNF, '-o',label="VNF")
plt.plot(thicknesses, deformation_FPU, '-o',label="FPF")
plt.plot(thicknesses, deformation_SBT, '-o',label="SBT")
plt.plot(thicknesses, deformation_MFM, '-o',label="MYF")
plt.plot(thicknesses, deformation_RPF, '-o',label="RPF")
plt.xlabel("Thickness (cm)")
plt.ylabel("Deformation (mm)")
plt.title("Deformation vs Thickness")
plt.grid(True, linestyle=':', linewidth=0.7, color='lightgray')
plt.legend()
plt.grid(True)
plt.show()

# Acceleration vs Thickness
plt.figure(figsize=(6, 5))
plt.plot(thicknesses, acceleration_PXD, '-o',label="PXD")
plt.plot(thicknesses, acceleration_EVA, '-o',label="EVA")
plt.plot(thicknesses, acceleration_VNF, '-o',label="VNF")
plt.plot(thicknesses, acceleration_FPU, '-o',label="FPF")
plt.plot(thicknesses, acceleration_SBT, '-o',label="SBT")
plt.plot(thicknesses, acceleration_MFM, '-o',label="MYF")
plt.plot(thicknesses, acceleration_RPF, '-o',label="RPF")
plt.xlabel("Thickness (cm)")
plt.ylabel("Maximum Acceleration ($ms^{-2}$)")
plt.title("Acceleration vs Thickness")
plt.legend()
plt.grid(True)
plt.show()


# Stress vs Thickness
plt.figure(figsize=(6, 5))
plt.plot(thicknesses, stress_PXD, '-o',label="PXD")
plt.plot(thicknesses, stress_EVA, '-o',label="EVA")
plt.plot(thicknesses, stress_VNF, '-o',label="VNF")
plt.plot(thicknesses, stress_FPU, '-o',label="FPF")
plt.plot(thicknesses, stress_SBT, '-o',label="SBT")
plt.plot(thicknesses, stress_MFM, '-o',label="MYF")
plt.plot(thicknesses, stress_RPF, '-o',label="RPF")
plt.xlabel("Thickness (cm)")
plt.ylabel("Stress (kPa)")
plt.title("Stress vs Thickness")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 5))
plt.plot(thicknesses, acceleration_PXD, '-o',label="PXD")
plt.plot(thicknesses, acceleration_EVA, '-o',label="EVA")
plt.plot(thicknesses, acceleration_VNF, '-o',label="VNF")
plt.plot(thicknesses, acceleration_SBT, '-o',label="SBT")
plt.plot(thicknesses, acceleration_MFM, '-o',label="MFM")
plt.xlabel("Thickness (mm)")
plt.ylabel("Acceleration ($ms^{-2}$)")
plt.title("Max Acceleration vs Thickness")
plt.legend()
plt.grid(True)
plt.show()
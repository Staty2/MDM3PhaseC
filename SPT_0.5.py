import matplotlib.pyplot as plt
import numpy as np

# Given EVA data
time = np.array([
    0, 0.000150627, 0.000300427, 0.000450227, 0.000600023, 0.000750679, 0.000900475,
    0.00105027, 0.00120006, 0.00135072, 0.00150052, 0.00165031, 0.00180011,
    0.00195076, 0.00210058, 0.00225039, 0.00240021, 0.00255002, 0.0027007,
    0.00285051, 0.003
])

deformation_eva = np.array([0.000000,0.000055,0.000110,0.000221,0.000552,0.001104,0.005520,0.011040,
                            0.022080,0.055200,0.088320,0.132480,0.198720,0.276000,0.353280,0.430560,
                            0.485760,0.518880,0.540960,0.546480,0.552000])

stress_eva = np.array([0.000000,0.066699,0.133398,0.156075,0.001467,0.026079,0.196695,0.899900,2.928611,7.402633,
                       15.116549,26.006729,38.895472,52.108638,63.931200,73.440715,80.198637,84.698138,
                       89.051569,96.464473,109.320000])

acceleration_eva = np.array([0.000000,0.000000,0.004099,0.020497,0.015988,0.258103,1.762203,7.213856,
                             20.605630,44.393277,74.278491,99.248429,108.259087,99.129544,79.464335,57.413228,
                             36.013528,20.094835,24.488248,61.869363,120.000000])

# Create separate figures for each plot
plt.figure(figsize=(8, 5))
plt.plot(time, deformation_eva, marker='o', linestyle='-', color='b', label="Deformation")
plt.xlabel("Time (s)")
plt.ylabel("Deformation (m)")
plt.title("Deformation vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, stress_eva, marker='s', linestyle='-', color='r', label="Von Mises Stress")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.title("Von Mises Stress vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, acceleration_eva, marker='^', linestyle='-', color='g', label="Acceleration")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

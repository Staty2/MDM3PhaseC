import matplotlib.pyplot as plt
import numpy as np

# EVA data for 0.5cm thickness
time_05cm = [
    0, 0.000150627, 0.000300427, 0.000450227, 0.000600023, 0.000750679, 0.000900475,
    0.00105027, 0.00120006, 0.00135072, 0.00150052, 0.00165031, 0.00180011,
    0.00195076, 0.00210058, 0.00225039, 0.00240021, 0.00255002, 0.0027007,
    0.00285051, 0.003
]
deformation_05cm = [
    0, 0, 0, 8.03E-15, 2.91E-11, 1.83E-09, 2.31E-08, 1.58E-07, 7.25E-07,
    2.48E-06, 6.70E-06, 1.50E-05, 2.86E-05, 4.82E-05, 7.30E-05, 0.000102225,
    0.000134577, 0.000168922, 0.000204666, 0.000241706, 0.000282137
]
stress_05cm = [
    0, 0, 1.96E-09, 2.34E-05, 0.00220191, 0.0391433, 0.294956, 1.34923,
    4.39083, 11.0986, 22.6639, 38.9913, 58.3151, 78.1253, 95.8506, 110.108,
    120.242, 126.986, 133.513, 144.627, 163.901
]
acceleration_05cm = [
    0, 0, 5.25E-08, 0.00046844, 0.039026, 0.629653, 4.29861, 17.5974,
    50.2647, 108.295, 181.191, 242.103, 264.083, 241.819, 193.846,
    140.05, 87.8496, 49.0181, 59.7357, 150.921, 292.728
]

# EVA data for 1cm thickness
time_1cm = [
    0.00000000, 0.00015050, 0.00030030, 0.00045010, 0.00060070, 0.00075050,
    0.00090030, 0.00105000, 0.00120000, 0.00135000, 0.00150000, 0.00165000,
    0.00180000, 0.00195000, 0.00210000, 0.00225000, 0.00240000, 0.00255000,
    0.00270000, 0.00285000, 0.00300000
]
deformation_1cm = [0,0,0.00E+00,1.96E-16,9.03E-14,1.84E-10,4.08E-09,3.82E-08,2.16E-07,8.43E-07,2.50E-06,5.99E-06,
                   1.21E-05,2.14E-05,3.39E-05,4.94E-05,6.77E-05,8.86E-05,0.000111711,0.000136742,0.000163418]
stress_1cm = [0,0.00E+00,1.05E-11,7.36E-07,0.000183108,0.00602075,0.0684633,0.403225,1.54264,4.28894,9.42921,17.2251,27.1614,38.2751,
              49.4464,60.1484,70.1039,79.171,87.1565,94.3643,101.591]
acceleration_1cm = [0,0.00E+00,3.05E-10,1.55E-05,0.00339707,0.101958,1.05809,5.60999,18.8955,45.0075,81.8247,118.336,
                    140.882,144.074,134.707,123.533,113.915,101.604,85.4737,73.7424,78.8764]

# EVA data for 2cm thickness
time_2cm = [
    0.00000000, 0.00015030, 0.00030010, 0.00045070, 0.00060050, 0.00075030,
    0.00090010, 0.00105000, 0.00120000, 0.00135000, 0.00150000, 0.00165000,
    0.00180000, 0.00195000, 0.00210000, 0.00225000, 0.00240000, 0.00255000,
    0.00270000, 0.00285000, 0.00300000
]
deformation_2cm = [0,0,0,0,0,4.25E-15,3.24E-13,9.13E-10,1.20E-08,8.38E-08,3.86E-07,1.30E-06,3.41E-06,7.28E-06,1.32E-05,
                   2.08E-05,2.98E-05,3.97E-05,5.07E-05,6.31E-05,7.71E-05]
stress_2cm = [0,0,2.20E-19,1.37E-11,8.89E-08,2.42E-05,0.00119453,0.0199603,0.155221,0.719668,2.30353,5.56261,10.7475,
              17.1668,23.6008,29.0514,33.5136,37.7278,42.46,47.7917,53.3845]
acceleration_2cm = [0,0,8.14E-18,3.34E-10,1.86E-06,0.000459771,0.0209134,0.322088,2.28167,9.44351,26.223,52.7551,
                    79.9161,91.8661,80.5627,56.2117,40.8033,46.8466,63.9499,72.5197,67.4062]

# EVA data for 3cm thickness
time_3cm = [
    0.00000000, 0.00015000, 0.00030070, 0.00045050, 0.00060030, 0.00075010,
    0.00090080, 0.00105000, 0.00120000, 0.00135000, 0.00150000, 0.00165000,
    0.00180000, 0.00195000, 0.00210000, 0.00225000, 0.00240000, 0.00255000,
    0.00270000, 0.00285000, 0.00300000
]
deformation_3cm = [0,0,0,0,0,0,0,1.78E-14,1.78E-10,3.51E-09,3.06E-08,1.68E-07,6.50E-07,1.92E-06,4.55E-06,
                   9.00E-06,1.52E-05,2.27E-05,3.09E-05,3.93E-05,4.78E-05]
stress_3cm = [0,0,0,1.10E-16,1.19E-11,2.23E-08,4.94E-06,0.000267159,0.00546447,0.0540818,0.312369,1.18482,3.28842,
              7.11558,12.5301,18.5878,23.8315,27.3735,29.3386,30.7658,32.8679]
acceleration_3cm = [0,0,0,2.99E-15,2.72E-10,4.58E-07,9.39E-05,0.00474326,0.0905123,0.827316,4.33361,14.5503,
                    34.5573,60.9761,81.4855,81.734,58.6999,26.6499,6.92746,10.8894,32.2447]

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Create separate figure for deformation
fig_deform = plt.figure(figsize=(12, 7))
plt.plot(time_05cm, deformation_05cm, marker='o', markersize=6, linestyle='-', linewidth=2, color='#1f77b4', label="0.5 cm")
plt.plot(time_1cm, deformation_1cm, marker='s', markersize=6, linestyle='-', linewidth=2, color='#ff7f0e', label="1.0 cm")
plt.plot(time_2cm, deformation_2cm, marker='^', markersize=6, linestyle='-', linewidth=2, color='#2ca02c', label="2.0 cm")
plt.plot(time_3cm, deformation_3cm, marker='d', markersize=6, linestyle='-', linewidth=2, color='#d62728', label="3.0 cm")
plt.xlabel("Time (s)")
plt.ylabel("Deformation (m)")
plt.title("Deformation vs Time - EVA Foam at Different Thicknesses")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eva_deformation_comparison.png', dpi=300, bbox_inches='tight')

# Create separate figure for stress
fig_stress = plt.figure(figsize=(12, 7))
plt.plot(time_05cm, stress_05cm, marker='o', markersize=6, linestyle='-', linewidth=2, color='#1f77b4', label="0.5 cm")
plt.plot(time_1cm, stress_1cm, marker='s', markersize=6, linestyle='-', linewidth=2, color='#ff7f0e', label="1.0 cm")
plt.plot(time_2cm, stress_2cm, marker='^', markersize=6, linestyle='-', linewidth=2, color='#2ca02c', label="2.0 cm")
plt.plot(time_3cm, stress_3cm, marker='d', markersize=6, linestyle='-', linewidth=2, color='#d62728', label="3.0 cm")
plt.xlabel("Time (s)")
plt.ylabel("Von Mises Stress (kPa)")
plt.title("Von Mises Stress vs Time - EVA Foam at Different Thicknesses")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eva_stress_comparison.png', dpi=300, bbox_inches='tight')

# Create separate figure for acceleration
fig_accel = plt.figure(figsize=(12, 7))
plt.plot(time_05cm, acceleration_05cm, marker='o', markersize=6, linestyle='-', linewidth=2, color='#1f77b4', label="0.5 cm")
plt.plot(time_1cm, acceleration_1cm, marker='s', markersize=6, linestyle='-', linewidth=2, color='#ff7f0e', label="1.0 cm")
plt.plot(time_2cm, acceleration_2cm, marker='^', markersize=6, linestyle='-', linewidth=2, color='#2ca02c', label="2.0 cm")
plt.plot(time_3cm, acceleration_3cm, marker='d', markersize=6, linestyle='-', linewidth=2, color='#d62728', label="3.0 cm")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Time - EVA Foam at Different Thicknesses")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('eva_acceleration_comparison.png', dpi=300, bbox_inches='tight')

plt.show()
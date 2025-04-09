import matplotlib.pyplot as plt

# Given EVA data
time = [
    0, 0.000150627, 0.000300427, 0.000450227, 0.000600023, 0.000750679, 0.000900475,
    0.00105027, 0.00120006, 0.00135072, 0.00150052, 0.00165031, 0.00180011,
    0.00195076, 0.00210058, 0.00225039, 0.00240021, 0.00255002, 0.0027007,
    0.00285051, 0.003
]
deformation = [0,0,1.37E-18,5.35E-14,2.69E-10,8.17E-09,9.24E-08,5.92E-07,2.55E-06,8.14E-06,2.02E-05,
               4.07E-05,6.99E-05,0.000105707,0.000144718,0.00018491,0.000225391,0.00026701,0.000313473,0.000369857,0.000441368]
stress = [0,0,1.50E-08,0.000147648,0.010465,0.15978,1.10886,4.73819,14.2316,32.3744,58.2186,87.2189,113.117,131.829,
          142.369,147.002,151.229,163.843,192.865,238.785,293.157]
acceleration = [0,0,4.03E-07,0.00291681,0.182887,2.53645,15.9293,60.5195,156.487,289.684,390.761,391.792,287.989,
                148.628,47.647,7.17598,41.1722,192.444,455.084,695.865,741.959]

# Create separate figures for each plot
plt.figure(figsize=(8, 5))
plt.plot(time, deformation, marker='o', linestyle='-', color='b', label="Deformation")
plt.xlabel("Time (s)")
plt.ylabel("Deformation (m)")
plt.title("Deformation vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, stress, marker='s', linestyle='-', color='r', label="Von Mises Stress")
plt.xlabel("Time (s)")
plt.ylabel("Stress (MPa)")
plt.title("Von Mises Stress vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(time, acceleration, marker='^', linestyle='-', color='g', label="Acceleration")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration vs Time for EVA")
plt.legend()
plt.grid()
plt.show()

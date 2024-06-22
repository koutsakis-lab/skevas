import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import requests

# Load the Analytical solution .csv created in WebPlotDigitizer fig.3-43/p. 470 of the Nellis book pdf

# Click on the "Raw" button to get the raw URL (github.com) of the file.
url = 'https://raw.githubusercontent.com/koutsakis-lab/skevas/main/analytical_plot.csv?token=GHSAT0AAAAAACT5TB3A4KAHZAGDCV4Z4746ZTVXOXQ'
response = requests.get(url)

if response.status_code == 200:
    data = response.text
    print(data)
else:
    print('Failed to fetch the file:', response.status_code)

data_io = StringIO(data)
df_extracted = pd.read_csv(data_io, header=None)

# Rename columns for clarity
df_extracted.columns = ['x', 'T']

# Geometric and material parameters
L = 0.05  # m chamber wall thickness
rho = 2000  # kg/m^3 density
c = 200  # J/kgK specific heat capacity
k = 5  # W/mK thermal conductivity
a = k / (rho * c)  # m^2/s thermal diffusivity constant

# Constants given in the document
Tinitial = 293  # Initial temperature in K
Tinf = 473  # Fluid temperature in K
h = 500  # Heat transfer coefficient in W/m^2-K
Bi = h * L / k  # Biot number

# Grid and time constants
N = 51  # number of nodes
dx = L / (N - 1)  # spatial step
domain = np.linspace(0, L, N)
dt = 0.03  # time step
total_time = 600  # s total time duration
time_array = np.arange(0, total_time + dt, dt)
cfl = a * dt / dx**2
print("CFL:", cfl)

# Numerical solution (Euler method, p.468 of the Nellis book pdf)
# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

# Compute the numerical solution
for t in range(len(time_array) - 1):
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2)
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2) + (2 * h * dt / (rho * c)) * (Tinf - T_numerical[i, t]) / dx

# Plot the extracted analytical data
plt.figure(figsize=(10, 6))
plt.scatter(df_extracted['x'], df_extracted['T'], label='Analytical Solution', marker='o')
plt.xlabel('Axial Position (x)')
plt.ylabel('Temperature (K)')
plt.title('Analytical Solution')
plt.legend(loc='best')
plt.grid(True)

# Time steps to plot: 0, 4, 10, 20, 30, 40 seconds and the last time step
selected_times = [0, 4, 10, 20, 30, 40, 600]

# Find the closest index in time_array for each selected time
selected_indices = [np.abs(time_array - sec).argmin() for sec in selected_times]

# Plot numerical solution for selected time steps
plt.figure(figsize=(8, 6))
for idx in selected_indices:
    sec = time_array[idx]
    plt.plot(domain, T_numerical[:, idx], label=f'Time={sec:.1f}s')
plt.title('Numerical Temperature Distribution (Euler)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

# Plot comparative graph
plt.figure(figsize=(8, 6))
for idx in selected_indices:
    sec = time_array[idx]
    plt.plot(domain, T_numerical[:, idx], label=f'Time={sec:.1f}s')
plt.scatter(df_extracted['x'], df_extracted['T'], label='Analytical Solution', marker='o')
plt.title('Numerical-Analytical comparative graph')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)
plt.show()
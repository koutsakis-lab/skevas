import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Geometric and material parameters
L = 0.5  # m chamber wall thickness
rho = 8900  # kg/m^3 density
c = 390  # J/kgK specific heat capacity
k = 320  # W/mK thermal conductivity
a = k / (rho * c)  # m^2/s thermal diffusivity constant

# Constants given in the document
Tinitial = 293  # Initial temperature in K
q_initial = 15 * 10**6  # W/m^2

# Grid and time constants
N = 51  # number of nodes
dx = L / (N - 1)  # spatial step
domain = np.linspace(0, L, N)
dt = 0.03  # time step
total_time = 600  # s total time duration
time_array = np.arange(0, total_time + dt, dt)
cfl = a * dt / dx**2
print("CFL:", cfl)

# Load Tgas and Twall values from the .xlsx files
file_path_tgas = 'C:/Users/user/Desktop/python/Tgas_values.xlsx'
file_path_twall = 'C:/Users/user/Desktop/python/Twall_values.xlsx'

df_Tgas = pd.read_excel(file_path_tgas)
df_Twall = pd.read_excel(file_path_twall)

# Ensure time steps in Tgas and Twall match with time_array
if len(df_Tgas) > len(time_array):
    df_Tgas = df_Tgas.iloc[:len(time_array)]
elif len(df_Tgas) < len(time_array):
    df_Tgas = df_Tgas.reindex(range(len(time_array)), method='nearest')

if len(df_Twall) > len(time_array):
    df_Twall = df_Twall.iloc[:len(time_array)]
elif len(df_Twall) < len(time_array):
    df_Twall = df_Twall.reindex(range(len(time_array)), method='nearest')

# Initialize heat flux
q = np.zeros(len(time_array) - 1)
q[0] = q_initial  # Initial heat flux for the first time step

# Matrices for the cost function
B = 12 * np.eye(1)  # Background error covariance matrix
R = 50 * np.eye(1)  # Observation error covariance matrix

# Numerical solution (Euler method)
# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

# Function to compute the cost
def cost_function(x, xb, y0, Gx):
    x = np.array([x])
    xb = np.array([xb])
    y0 = np.array([y0])
    Gx = np.array([Gx])
    term1 = 0.5 * (x - xb).T @ np.linalg.inv(B) @ (x - xb)
    term2 = 0.5 * (y0 - Gx).T @ np.linalg.inv(R) @ (y0 - Gx)
    return term1 + term2

# Compute the numerical solution and optimize heat flux
for t in range(len(time_array) - 1):
    Tinf = df_Tgas['Tgas'][t]  # Set Tinf to the corresponding Tgas value for this time step
    
    # Update the numerical solution
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2)
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            # Apply the boundary condition at x = L
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2) + q[t] * dt / (rho * c * dx)
    
    # Observation and model output at the last node (x = L)
    y0 = df_Twall['Twall'][t]
    Gx = T_numerical[-1, t + 1]
    
    # Optimize the heat flux for the current time step
    res = minimize(cost_function, q[t], args=(q[t], y0, Gx), method='L-BFGS-B')
    
    # Update heat flux and store the result
    if t < len(time_array) - 2:
        q[t + 1] = res.x[0]  # Ensure we extract a single element from the array

# Plot numerical solution for selected time steps
selected_times = [0, 4, 10, 20, 30, 40, 600]
selected_indices = [np.abs(time_array - sec).argmin() for sec in selected_times]

plt.figure(figsize=(8, 6))
for idx in selected_indices:
    sec = time_array[idx]
    plt.plot(domain, T_numerical[:, idx], label=f'Time={sec:.1f}s')
plt.title('Numerical Temperature Distribution (Euler)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)

# Plot heat flux q at the first node over time
plt.figure(figsize=(8, 6))
plt.plot(time_array[:-1], q, label='Heat Flux at First Node')
plt.title('Heat Flux at First Node Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Heat Flux (W/m^2)')
plt.legend()
plt.grid(True)

plt.show()

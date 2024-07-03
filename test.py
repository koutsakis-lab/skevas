import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Geometric and material parameters
L = 0.15  # m chamber wall thickness
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
total_time = 60  # s total time duration
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
q = np.full(len(time_array) - 1, q_initial)  # Initialize with q_initial
q_calculated = np.zeros_like(q)

# Adjust matrices for the cost function
B = 1 * np.eye(1)  # Background error covariance matrix
R = 100 * np.eye(1)  # Observation error covariance matrix

# Numerical solution (Euler method)
# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

# Function to compute the cost
def cost_function(x, xb, y0, T0):
    x = np.array([x])
    xb = np.array([xb])
    y0 = np.array([y0])
    T0 = np.array([T0])
    term1 = 0.5 * (x - xb).T @ np.linalg.inv(B) @ (x - xb)
    term2 = 0.5 * (y0 - T0).T @ np.linalg.inv(R) @ (y0 - T0)
    return term1 + term2

# Compute the numerical solution and optimize heat flux
for t in range(len(time_array) - 1):
    T_numerical[N-1, t] = df_Tgas['Tgas'][t]  # Apply Tgas as the boundary condition at x = L

    # Update the numerical solution for the next time step
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2)
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            # Apply the boundary condition at x = L with heat flux
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2) + q[t] * dt / (rho * c * dx)
        q_calculated[t] = k * (T_numerical[N-1, t] - T_numerical[N-2, t]) / dx

    # Observation and model output at the first node (x = 0)
    y0 = df_Twall['Twall'][t]
    T0 = T_numerical[0, t + 1]  # Directly use the temperature at the first node (x = 0)

    # Optimize the heat flux for the current time step
    res = minimize(cost_function, q_calculated[t], args=(q_calculated[t], y0, T0), method='L-BFGS-B')

    # Print the cost function value for the first 30 time steps
    if t < 30:
        cost_value = cost_function(res.x[0], q_calculated[t], y0, T0)
        print(f"Time step {t+1}, Cost function value: {cost_value}, Optimized q: {res.x[0]}")

    # Update heat flux and store the result
    if t < len(time_array) - 2:
        q[t + 1] = res.x[0]  # Ensure we extract a single element from the array

# Plot numerical solution for selected time steps
selected_times = [0, 4, 10, 20, 30, 40]
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

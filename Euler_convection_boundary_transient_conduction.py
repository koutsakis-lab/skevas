import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Geometric parameters
L = 0.05  # m chamber wall thickness
rho = 2000  # kg/m^3 density
c = 200  # J/kgK specific heat capacity
k = 5  # W/mK thermal conductivity
a = k / (rho * c)  # m^2/s thermal diffusivity constant

# Constants given in the document
Tinitial = 283  # Initial temperature in K
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

# Define the number of terms to sum in the series
num_terms = 20

# Define the position array
x_tilde = np.linspace(0, 1, 100)  # Normalized position from 0 to 1

# Define Fourier numbers based on the entire time array
Fo_values = (a * time_array) / L**2

# Function to find eigenvalues using the eigencondition
def eigencondition(zeta):
    return np.tan(zeta) - Bi / zeta

# Find the first 'num_terms' eigenvalues
eigenvalues = []
for i in range(1, num_terms + 1):
    zeta_guess = (i - 0.5) * np.pi
    zeta_i = fsolve(eigencondition, zeta_guess)[0]
    eigenvalues.append(zeta_i)

# Calculate Ci for each term
def Ci(zeta_i):
    return 2 * np.sin(zeta_i) / (zeta_i + np.sin(zeta_i) * np.cos(zeta_i))

# Initialize theta array for each Fourier number
T_analytical = np.zeros((len(time_array), len(x_tilde)))

# Sum the series for each specific Fourier number
for t_idx, Fo in enumerate(Fo_values):
    if t_idx == 0:
        T_analytical[t_idx] = Tinitial
    else:
        theta = np.zeros_like(x_tilde)
        for zeta_i in eigenvalues:
            Ci_val = Ci(zeta_i)
            theta += Ci_val * np.cos(zeta_i * x_tilde) * np.exp(- (zeta_i)**2 * Fo)
        T_analytical[t_idx] = theta + Tinf  # Use theta to calculate the temperature

# Numerical solution (Euler, p. 468 Nellis pdf)

# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

for t in range(len(time_array) - 1):
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2)
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2) + (2 * h * dt / (rho * c)) * (Tinf - T_numerical[i, t]) / dx

# Selected time steps for plotting
selected_times = [20, 50, 100, 200, 400, 600]
selected_indices = [np.abs(time_array - t).argmin() for t in selected_times]

# Plot the analytical results
plt.figure(figsize=(8, 6))
for idx in selected_indices:
    time = time_array[idx]
    plt.plot(x_tilde * L, T_analytical[idx], label=f'Time={time:.1f}s, Fo={Fo_values[idx]:.3f}')

plt.title('Temperature Distribution - Analytical Solution')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot numerical solution for selected time steps
plt.figure(figsize=(8, 6))
for idx in selected_indices:
    time = time_array[idx]
    plt.plot(domain, T_numerical[:, idx], label=f'Time={time:.1f}s')

plt.title('Numerical Temperature Distribution (Euler)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot both analytical and numerical solutions for comparison
plt.figure(figsize=(8, 6))
for idx in selected_indices:
    time = time_array[idx]
    plt.plot(domain, T_numerical[:, idx], 'o', label=f'Numerical Time={time:.1f}s')
    plt.plot(x_tilde * L, T_analytical[idx], '-', label=f'Analytical Time={time:.1f}s')

plt.title('Comparison of Analytical and Numerical Solutions')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True)
plt.show()

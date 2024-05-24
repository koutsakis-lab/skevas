import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Geometric parameters
L = 0.2  # m chamber wall thickness
A = 1  # m^2 arbitrary surface area
rho = 8900  # kg/m^3 density, GRCop42
c = 390  # J/kgK specific heat capacity GRCop42
k = 320  # W/mK thermal conductivity GRCop42 320
a = k / (rho * c)  # m^2/s thermal diffusivity constant

q = 5e5  # W/m^2 heat flux
Tinitial = 500  # K initial temperature

# Grid and time constants
N = 51  # number of nodes
dx = L / (N - 1)  # spatial step
domain = np.linspace(0, L, N)
dt = 1e-4  # time step
total_time = 600  # s total time duration
fractions = np.linspace(0, 5, N)
time_intervals = fractions * total_time  # time intervals in seconds
time_array = np.arange(0, total_time + dt, dt)

## Analytical solution

stored_profiles = {t: None for t in time_intervals}

# Loop 
for t in time_array:
    T_analytical = np.zeros_like(domain)
    if t == 0:
        T_analytical.fill(Tinitial)
    elif t > 0:
        for i, x in enumerate(domain):
            T_analytical[i] = Tinitial + (q / k) * (
                (np.sqrt((4 * a * t) / np.pi)) * (np.exp((-x ** 2) / (4 * a * t))) - x * special.erfc(x / (np.sqrt(4 * a * t))))  # Nellis-Klein p.398 of the pdf
    
    if t in time_intervals:
        stored_profiles[t] = T_analytical.copy()

## Numerical solution

# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0
aa = np.zeros(len(domain))
bb = np.zeros(len(domain))
cc = np.zeros(len(domain))
dd = np.zeros(len(domain))
T_cap = np.zeros(len(domain))
T_double_cap = np.zeros(len(domain))

# Loop (Runge-Kutta method) p. 362 of the pdf 

for t in range(len(time_array) - 1):
    for i, x in enumerate(domain):
        if i == 0:
            aa[i] = 2 * q / (A * rho * c * dx) + 2 * a * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx ** 2)
        elif 0 < i < N - 1:
            aa[i] = a * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx ** 2)
        elif i == N - 1:
            aa[i] = 2 * a * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx ** 2)
        T_cap[i] = T_numerical[i, t] + aa[i] * dt / 2
        
    for i, x in enumerate(domain):
        if i == 0:
            bb[i] = 2 * q / (A * rho * c * dx) + 2 * a * (T_cap[i + 1] - T_cap[i]) / (dx ** 2)
        elif 0 < i < N - 1:
            bb[i] = a * (T_cap[i + 1] - 2 * T_cap[i] + T_cap[i - 1]) / (dx ** 2)
        elif i == N - 1:
            bb[i] = 2 * a * (T_cap[i - 1] - T_cap[i]) / (dx ** 2)
        T_double_cap[i] = T_numerical[i, t] + bb[i] * dt / 2
        
    for i, x in enumerate(domain):
        if i == 0:
            cc[i] = 2 * q / (A * rho * c * dx) + 2 * a * (T_double_cap[i + 1] - T_double_cap[i]) / (dx ** 2)
        elif 0 < i < N - 1:
            cc[i] = a * (T_double_cap[i + 1] - 2 * T_double_cap[i] + T_double_cap[i - 1]) / (dx ** 2)
        elif i == N - 1:
            cc[i] = 2 * a * (T_double_cap[i - 1] - T_double_cap[i]) / (dx ** 2)
        T_cap[i] = T_numerical[i, t] + cc[i] * dt
        
    for i, x in enumerate(domain):
        if i == 0:
            dd[i] = 2 * q / (A * rho * c * dx) + 2 * a * (T_cap[i + 1] - T_cap[i]) / (dx ** 2)
        elif 0 < i < N - 1:
            dd[i] = a * (T_cap[i + 1] - 2 * T_cap[i] + T_cap[i - 1]) / (dx ** 2)
        elif i == N - 1:
            dd[i] = 2 * a * (T_cap[i - 1] - T_cap[i]) / (dx ** 2)
        T_numerical[i, t + 1] = T_numerical[i, t] + (aa[i] + 2 * bb[i] + 2 * cc[i] + dd[i]) * dt / 6

##calculate error

# Calculate error between the last time step values for T_analytical and T_numerical

error = np.zeros(len(domain))
T_analytical_last = stored_profiles[total_time]
T_numerical_last = T_numerical[:, -1]

for i in range(len(domain)):
    error[i] = np.abs((T_analytical_last[i] - T_numerical_last[i])) / T_numerical_last[i] * 100

# Calculate the maximum percentage error

max_error = np.max(error)

print("Maximum percentage error:", max_error, "%")

# Plot the analytical solution

plt.figure(figsize=(10, 6))
for t, profile in stored_profiles.items():
    if profile is not None:
        plt.plot(domain, profile, label=f't = {t:.2f} s')

plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Transient Heat Conduction - Analytical Solution')
plt.legend()
plt.grid(True)

# Plot the numerical solution for the last time step
plt.figure()
plt.plot(domain, T_numerical[:, -1], label="Numerical Solution")
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Distribution at Final Time Step')
plt.legend()
plt.show()

# Plot the analytical and numerical solution at the final time step

plt.figure()
plt.plot(domain, stored_profiles[total_time], label="Analytical Solution", linestyle='-', color='blue')
plt.plot(domain, T_numerical[:, -1], label="Numerical Solution", linestyle='--', color='red')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.title('Comparison of Analytical and Numerical Solutions at Final Time Step')
plt.legend()
plt.grid(True)

plt.show()





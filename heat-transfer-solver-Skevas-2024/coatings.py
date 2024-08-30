#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sourceCode.responseFunctions import wallTimeResponse
from sourceCode.BuildSurfaceHeatFlux import HeatFlux
import time
tic=time.time()

# %% Import properties
from MultilayerArchitecturesLibrary.importWallData import multilayerPropertiesArchitecture

iWallCase=2
materialDatabase = multilayerPropertiesArchitecture(iWallCase)

k=np.array(materialDatabase.ThermalConductivity)
rhoc=np.array(materialDatabase.Density \
            * materialDatabase.SpecificHeatCapacity)
# fix depending on where you want to measure temperature
L=np.array(materialDatabase.Thickness)
E = np.array(materialDatabase.YoungsModulus)
sigma = np.array(materialDatabase.Stress)
print("E:", E, "Pa")
print("sigma:", sigma, "Pa")
R=L/k
C=rhoc*L

a = k / rhoc  # m^2/s thermal diffusivity constant

thermal_shock_resistance = (k*sigma)/(a*E)
chamber_pressure = 35*101325 #Pa
print("maximum chamber pressure:", chamber_pressure, "Pa")
print("thermal shock resistance parameter:", thermal_shock_resistance)
Delta_L = 500e-6 #m expansion (arbitrary)


# Constants given in the document
Tinitial = 293  # Initial temperature in K


# Grid and time constants
N = 11  # number of nodes
dx = L / (N - 1)  # spatial step
domain = np.linspace(0, L, N)


# %% Setup time domain
resWindowCA = 1 # [deg] Resolution window for highest heat flux frequency component
EngRPM = 3e03 # [rpm] Engine speed
fCycle = EngRPM/60 # [Hz] Cyclic frequency

fSignal = fCycle * 360/resWindowCA # [Hz] Highest heat flux signal frequency component
n = 2
fSampling = n * fSignal # [Hz] Nyquist Sampling frequency
numHarmonics = int(fSignal/fCycle)
ωSampling = 2*np.pi*fCycle * np.linspace(0,numHarmonics,numHarmonics,dtype=int)
Δ=1/fSampling
dt = Δ  # time step
print("dt:", 1000*dt, "ms")
# %%
numCycles = 1 # [-] Number of cycles
θCycle=1/fCycle # [s] Cycle period
θ=np.arange(0,θCycle*numCycles,Δ)
sizeOfθ=len(θ)
sizeOfθCycle=int(sizeOfθ/numCycles)
# %% Heat Flux
q_prime=16e6 # [W/m^2] applied heat flux at surface
f=numCycles # [Hz] cyclic frequency
T_backside=500 # [K] Coolant temperature
A_s=1 #[m^2] surface area
total_time = θ[-1]  # Use the last element of θ as the total time duration
time_array = np.arange(0, total_time + dt, dt)
cfl = a * dt / dx**2
material_time_constant = (L**2)/a
print("CFL:", cfl)
print("material time constant:", material_time_constant, "s") 

qFlux=HeatFlux(θ/(θCycle*numCycles),f,q_prime,A_s,profile='gauss')

# %% 

iInterface = 0
locOfX=[0]
X= wallTimeResponse(Δ,sizeOfθCycle,R,C,locOfX)

# Calculate the number of engine cycles required to attenuate the response below the tolerance threshold
numCycles = int(np.size(X[iInterface])/sizeOfθCycle) # [-] Number of cycles
sizeOfθ = np.size(X[iInterface])

q_guess = 16e6 # [W/m2] Initial heat flux estimate
T_initialEffect = q_guess*(np.sum(R[iInterface:])-np.cumsum(X[iInterface]))

T = np.convolve(np.tile(qFlux, numCycles), X[iInterface])[:sizeOfθ] + T_backside + T_initialEffect
T = T[-sizeOfθCycle:]

# store sustained response of a single (last) cycle
Ymeasured = T # recorded surface temperature data
Ymeasured = np.tile(Ymeasured, numCycles)


# Numerical solution (Euler method)
# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

# Compute the numerical solution and optimize heat flux
for t in range(len(time_array) - 1):
    T_numerical[0, t] = T[t]   # Apply surface temperature as the boundary condition at x = 0

    # Update the numerical solution for the next time step
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2) + qFlux[t] * dt / (rhoc * dx)  
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            # Apply the boundary condition at x = L with heat flux
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2)

# Calculate the thermal expansion coefficient
min_length = min(len(T), len(T_numerical[N-1, :]))
T = T[:min_length]
T_numerical_N1 = T_numerical[N-1, :min_length]

Delta_T = np.argmin(T - T_numerical_N1)


CTE = Delta_L / (L * Delta_T)
print("Thermal expansion coefficient:", CTE, "1/K")

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot heat flux on the left y-axis
ax1.plot(θ, qFlux, '--k', label='Heat Flux')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Heat Flux [W/m²]', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Create a second y-axis for the temperatures
ax2 = ax1.twinx()
ax2.plot(θ, T, '-k', label='Surface Temperature')
ax2.plot(time_array, T_numerical[N-1, :], '-r', label='Substrate Temperature')
ax2.set_ylabel('Temperature [K]', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Adding legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot
plt.title('Heat Flux and Temperature over Time')
plt.grid(True)

plt.show()
# %% 
toc=time.time(); 
print('Elapsed time:',toc-tic, 'seconds')
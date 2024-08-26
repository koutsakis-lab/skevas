import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sourceCode.responseFunctions import wallTimeResponse
from sourceCode.BuildSurfaceHeatFlux import HeatFlux
from io import StringIO

def fetch_csv_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.text
        data_io = StringIO(data)
        df = pd.read_csv(data_io)
        return df
    else:
        print('Failed to fetch the file:', response.status_code)
        return None

# URLs for Tgas and Twall values (Replace with the actual raw URLs of your CSV files)
url_tgas = 'https://raw.githubusercontent.com/koutsakis-lab/skevas/main/heat-transfer-solver-Skevas-2024/Temperature_vs_Time_csv.csv'
url_qFlux = 'https://raw.githubusercontent.com/koutsakis-lab/skevas/main/heat-transfer-solver-Skevas-2024/HeatFlux_vs_Time_csv.csv'

# Fetch the data
df_Tgas = fetch_csv_data(url_tgas)
df_qFlux = fetch_csv_data(url_qFlux)

if df_Tgas is None or df_Twall is None:
    raise ValueError("Failed to fetch one or both of the required data files.")

# %% Import properties
from MultilayerArchitecturesLibrary.importWallData import multilayerPropertiesArchitecture

iWallCase = 3
materialDatabase = multilayerPropertiesArchitecture(iWallCase)

# Assuming the arrays have single values, we extract them as scalars.
k = np.array(materialDatabase.ThermalConductivity).item()
rhoc = np.array(materialDatabase.Density * materialDatabase.SpecificHeatCapacity).item()
L = np.array(materialDatabase.Thickness).item()

R = L / k
C = rhoc * L

a = k / rhoc  # m^2/s thermal diffusivity constant

# Constants given in the document
Tinitial = 293  # Initial temperature in K


# Grid and time constants
N = 11  # number of nodes
dx = L / (N - 1)  # spatial step
domain = np.linspace(0, L, N)

resWindowCA = 1  # [deg] Resolution window for highest heat flux frequency component
EngRPM = 3e03  # [rpm] Engine speed
fCycle = EngRPM / 60  # [Hz] Cyclic frequency

fSignal = fCycle * 360 / resWindowCA  # [Hz] Highest heat flux signal frequency component
n = 2
fSampling = n * fSignal  # [Hz] Nyquist Sampling frequency
numHarmonics = int(fSignal / fCycle)
ωSampling = 2 * np.pi * fCycle * np.linspace(0, numHarmonics, numHarmonics, dtype=int)
Δ = 1 / fSampling
dt = Δ  # time step
numCycles = 1  # [-] Number of cycles
θCycle = 1 / fCycle  # [s] Cycle period
θ = np.arange(0, θCycle * numCycles, Δ)
sizeOfθ = len(θ)
sizeOfθCycle = int(sizeOfθ / numCycles)
total_time = θ[-1]  # Use the last element of θ as the total time duration
time_array = np.arange(0, total_time + dt, dt)
cfl = a * dt / dx**2
print("CFL:", cfl)

# Load Tgas and Twall values from the .xlsx files
file_path_tgas = 'C:/Users/user/Documents/GitHub/skevas/heat-transfer-solver-Skevas-2024/Temperature_vs_Time.xlsx'

file_path_qFlux = 'C:/Users/user/Documents/GitHub/skevas/heat-transfer-solver-Skevas-2024/HeatFlux_vs_Time.xlsx'

df_Tgas = pd.read_excel(file_path_tgas)

df_qFlux = pd.read_excel(file_path_qFlux)

# Ensure time steps in Tgas and Twall match with time_array
if len(df_Tgas) > len(time_array):
    df_Tgas = df_Tgas.iloc[:len(time_array)]
elif len(df_Tgas) < len(time_array):
    df_Tgas = df_Tgas.reindex(range(len(time_array)), method='nearest')



if len(df_qFlux) > len(time_array):
    df_qFlux = df_qFlux.iloc[:len(time_array)]
elif len(df_qFlux) < len(time_array):
    df_qFlux = df_qFlux.reindex(range(len(time_array)), method='nearest')

# Numerical solution (Euler method)
# Initialization
T_numerical = np.zeros((len(domain), len(time_array)))
T_numerical[:, 0] = Tinitial  # Set initial condition for t=0

# Compute the numerical solution and optimize heat flux
for t in range(len(time_array) - 1):
    T_numerical[N-1, t] = df_Tgas['Temperature (K)'][t]  # Apply Tgas as the boundary condition at x = L

    # Update the numerical solution for the next time step
    for i in range(N):
        if i == 0:
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i + 1, t] - T_numerical[i, t]) / (dx**2)
        elif 1 <= i < N - 1:
            T_numerical[i, t + 1] = T_numerical[i, t] + a * dt * (T_numerical[i + 1, t] - 2 * T_numerical[i, t] + T_numerical[i - 1, t]) / (dx**2)
        elif i == N - 1:
            # Apply the boundary condition at x = L with heat flux
            T_numerical[i, t + 1] = T_numerical[i, t] + 2 * a * dt * (T_numerical[i - 1, t] - T_numerical[i, t]) / (dx**2) + df_qFlux['Heat Flux (W/m^2)'][t] * dt / (rhoc * dx)

# Plot numerical solution for every 10th time step
plt.figure(figsize=(8, 6))
for t in range(0, len(time_array), 40):  # Step through time_array by 10
    plt.plot(domain, T_numerical[:, t])

plt.title('Numerical Temperature Distribution (Euler)')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (K)')
plt.grid(True)
plt.show()


plt.show()

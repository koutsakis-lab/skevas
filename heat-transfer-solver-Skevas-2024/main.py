#-*- coding: utf-8 -*-
import scipy.io
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sourceCode.responseFunctions import wallTimeResponse
from sourceCode.BuildSurfaceHeatFlux import HeatFlux
import time
tic=time.time()

# %% Import properties
from MultilayerArchitecturesLibrary.importWallData import multilayerPropertiesArchitecture

iWallCase=3
materialDatabase = multilayerPropertiesArchitecture(iWallCase)

k=np.array(materialDatabase.ThermalConductivity)
rhoc=np.array(materialDatabase.Density \
            * materialDatabase.SpecificHeatCapacity)
# fix depending on where you want to measure temperature
L=np.array(materialDatabase.Thickness)

R=L/k
C=rhoc*L
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

# Save qFlux to its own Excel file
qFlux_data = pd.DataFrame({'Time (s)': θ, 'Heat Flux (W/m^2)': qFlux})
qFlux_data.to_excel('HeatFlux_vs_Time.xlsx', index=False)

# Save T to its own Excel file
T_data = pd.DataFrame({'Time (s)': θ, 'Temperature (K)': T})
T_data.to_excel('Temperature_vs_Time.xlsx', index=False)

# Plot
fig, ax = plt.subplots()
ax.plot(θ, qFlux, '--k', label='heat flux')
ax.set_ylabel(r'Heat Flux [W/m2]')
ax.set_xlabel(r'Time [s]')

ax2 = ax.twinx()
ax2.plot(θ, T, '-k', label='surface temperature')
ax2.set_ylabel(r'Temperature [K]')


plt.show()
# %% 
toc=time.time(); 
print('Elapsed time:',toc-tic, 'seconds')
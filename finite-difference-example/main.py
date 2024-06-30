# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import BuildSurfaceHeatFlux

# %% Import properties
from MultilayerArchitecturesLibrary.importWallData import multilayerPropertiesArchitecture

iWallCase=0
materialDatabase = multilayerPropertiesArchitecture(iWallCase)

k=np.array(materialDatabase.ThermalConductivity)
rhoc=np.array(materialDatabase.Density \
            * materialDatabase.SpecificHeatCapacity)

L=np.array(materialDatabase.Thickness)

NpL=np.array([36]) # [-] Nodes per layer

# %% Import heat transfer data
from codeDependencies.FiniteDifferenceSolverConverged import FiniteDifferenceSolverConverged

# Setup time domain
resWindowCA = 12 # [deg] Resolution window for highest heat flux frequency component
EngRPM = 1200 # [rpm] Engine speed
fCycle = EngRPM/60 # [Hz] Cyclic frequency

fSignal = fCycle * 360/resWindowCA # [Hz] Highest heat flux signal frequency component
n = 2
fSampling = n * fSignal # [Hz] Nyquist Sampling frequency
numHarmonics = int(fSignal/fCycle)
ωSampling = 2*np.pi*fCycle * np.linspace(0,numHarmonics,numHarmonics,dtype=int)
Δ=1/fSampling

numCycles = 1 # [-] Number of cycles
θCycle=1/fCycle # [s] Cycle period
θ=np.arange(0,θCycle*numCycles,Δ)
sizeOfθ=len(θ)

# %% Heat Flux
q_prime=1e6                #[W/m^2] applied heat flux at surface
f=numCycles                # [Hz] cyclic frequency
Tc=0                     # [K] coolant temperature
A_s=1                      #[m^2] surface area

qFlux=BuildSurfaceHeatFlux.HeatFluxArtificial(θ/(θCycle*numCycles),f,q_prime,A_s,profile='triangular')

sizeOfθ=len(θ) # [-] Number of total time steps
TwSteady=273+20 # [K] Coolant temperature
totalNumElem=np.sum(NpL)# [-] Total number of elements in multilayer
Tw_FD, x = FiniteDifferenceSolverConverged(k,rhoc,L,NpL,Δ,sizeOfθ,TwSteady,qFlux)

# %% Figures
fig, ax = plt.subplots(1)

ax.plot(θ,Tw_FD[0]-273.15)

ax.legend()
ax.set_ylabel(r'Wall Temperature [$^{\circ}$C]')
ax.set_xlabel(r'Time [s]')

ax.xaxis.get_major_formatter()._usetex = False # important for x/y labels to 
ax.yaxis.get_major_formatter()._usetex = False


# -*- coding: utf-8 -*-
import numpy as np
from BuildMatrices import BuildMatrices
from scipy.sparse.linalg import spsolve

def FiniteDifferenceSolverConverged(k,rhoc,L,NpL,Δ,sizeOfθ,TwSteady,qFlux):
    h=1e10                     #[W/m^2-K] backside convection coefficient
    A_x=1                      #[m^2] conduction area
    A_s=1                      #[m^2] surface area
    layer=L.size

    # NpL= np.tile(NiL, layer)   # np.array([nodes,nodes,nodes,nodes]) # number of nodes per layer
    N=np.sum(NpL)              #total number of nodes
    dx=L/NpL                   #[m] node spacing array

    x=np.zeros(N+1)
    for i in range(layer):
        x[np.sum(NpL[0:i])+1:np.sum(NpL[0:i+1])+1]=np.cumsum(dx[i]*np.ones(NpL[i]))+x[np.sum(NpL[0:i])]
        
    C_CN,S,H = BuildMatrices(layer,k,rhoc,NpL,L,h,N,dx,A_x,A_s)
    
    A_CN = C_CN + S*Δ/2
    B_CN = C_CN - S*Δ/2
    
    Qs=np.zeros(N+1)
    t_inf=TwSteady             #[K] backside convection temperature
    tinf=np.zeros(N+1); tinf[N]=t_inf
    t_old=TwSteady*np.ones_like(Qs)# t_ss  #initialize temperature profile to steady state
    
    err=1e9
    threshold=1e-4             #solution tolerance
    ts_old=np.zeros(sizeOfθ)

    while err > threshold:
        ts=np.zeros(sizeOfθ)
        for i in range(sizeOfθ):
            Qs[0]=qFlux[i]
            r=Qs+H.dot(tinf)
            b_CN = r*Δ
            t_new=spsolve(A_CN,(B_CN*t_old + b_CN))
            ts[i]=t_new[0]
            t_old=t_new
        err = np.linalg.norm(ts-ts_old,np.inf)
        ts_old=ts
        # print(err)
    # met error criteria now calculate one more time storing all values
    temp=np.zeros((N+1,sizeOfθ))
    for i in range(sizeOfθ):
        Qs[0]=qFlux[i]
        r=Qs+H.dot(tinf)
        b_CN = r*Δ
        t_new=spsolve(A_CN,(B_CN*t_old + b_CN))
        temp[:,i]=t_new
        t_old=t_new
        
    return temp, x
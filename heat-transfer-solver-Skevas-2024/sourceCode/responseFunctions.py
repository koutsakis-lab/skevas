# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def wallTimeResponse(Δ,sizeOfθ,R,C,locOfX,absTol=1e-12):
    from sourceCode.rootFinder import search
    from sourceCode.TransferMatrix import Matrix, dMatrix
    eps=np.finfo(np.float64).eps # Smallest positive floating point in numpy

    # %% Tuning parameters
    # absTol=1e-12 # [[m^2-K/W]] absolute tolerance for response function calculation
    numPartitions=1 # to reduce RAM requirements
    # %% Below values are calculated only once, to generate the response functions X
    [βm,Bm,dDm]=search(Δ,R,C)
    
    βmΔ=βm*Δ # [-]

    [[_,_],[_, Do]]= Matrix(eps,R,C)
    [[_,_],[_,dDo]]=dMatrix(eps,R,C)
    
    # %% Initialize a list of X that contains all the requested responses at the locations defined from the iloc array
    # X[0] and X[i=1..len(locOfX)] represents the surface and all the interfaces, respectively.
    Xtot=[np.array([]) for i in range(max(locOfX)+1)]
    
    for iloc in locOfX:
        # Matrix coefficients limits at β->0
        [[_, Bo],[_,_]]= Matrix(eps,R[iloc:],C[iloc:])
        [[_,dBo],[_,_]]=dMatrix(eps,R[iloc:],C[iloc:])
    
        # For the interface calculations only. Replace Bm (from surface) to the interface one
        if iloc>0:    
            Bm=np.array([Matrix(βm[i],R[iloc:],C[iloc:])[0,1] for i in range(np.size(βm))])
        # Determine constants for each response function
        c0 = Bo/Do
        c1 =1/Δ*( dBo/Do- Bo*dDo/Do**2)
        Ω=Bm/(Δ*βm**2*dDm)
        term = Ω *(1-np.exp(βmΔ))**2
        # Combustion surface and interface response loop calculation (due to heat flux changes at the combustion surface)
        Xcurr = np.array([])
        
        # First two triangular-pulse-transformed values of the X response
        Xcurr = np.concatenate([Xcurr,[ c0 + c1 + Ω @ np.exp(-βmΔ)]]) 
        Xcurr = np.concatenate([Xcurr,[-c1 + Ω*(1-2*np.exp(βmΔ)) @ np.exp(-βmΔ*2)]]) 
        
        # Nested loop for the rest function
        iStart=iStop=2 # starts from index 2 since previous are computed already

        sizeOfSubArrayIndex=int(np.floor(sizeOfθ/numPartitions))
        subArrayXcurr = np.array([999]) # Dummy variable: a large number to be used just to read X_curr first time
    
        j=0
        while subArrayXcurr[-1] >= absTol:
            
            # Indexing
            iStart=iStop
            iStop+=sizeOfSubArrayIndex
        
            # Perform calculations to be used for every response function
            subArrayIndex=np.arange(iStart, iStop)
            exp_βmΔθ=np.exp(np.outer(-βmΔ,(subArrayIndex+1)))
            
            # Calculate response functions
            subArrayXcurr = term @ exp_βmΔθ
            Xcurr = np.concatenate([Xcurr,subArrayXcurr])
            
            j+=1 # partition counter
        
        # Pad to the closest end of the next closet cycle interval
        # nOf0sToBePadded = int(np.ceil(np.size(Xcurr)/sizeOfθ))*sizeOfθ - np.size(Xcurr)
        # Xcurr = np.pad(Xcurr,(0,nOf0sToBePadded), mode='constant')
        
        # Chop to the closest end of last cycle
        nOfIncompleteValuesOfLastCycle = np.size(Xcurr) - int(np.floor(np.size(Xcurr)/sizeOfθ))*sizeOfθ
        Xtot[iloc] = Xcurr[:-nOfIncompleteValuesOfLastCycle]
        
    return Xtot
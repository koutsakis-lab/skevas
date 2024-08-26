# -*- coding: utf-8 -*-
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %                                                                           %
# %          Root finding algorithm for Response Function Calculation         %
# %                              G. Koutsakis                                 %
# %                          Engine Research Center                           %
# %                   University of Wisconsin-Madison, USA                    %
# %                                                                           %
# %     Contact information: koutsakis@wisc.edu                               %
# %                          koutsakis.george@gmail.com                       %
# %                          +1 (608)-960-5208                                %
# %                                                                           %
# % Last modified: Nov. 5th, 2020                                             %
# %                                                                           %
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# In the following root finding algorithm the roots of the function D are found more efficiently than traditional rooting schemes.
# This algorithm is based on a kind of similar approach developed for building heat transfer applications,
# as shown in ref. [1]. In engines, the roots of D (and C) are of interest instead of roots of B (and A) for the buildings. 

# References:
# [1] Hittle, D.C. and Bishop, R., “An Improved Root-Finding Procedure for Use in Calculating Transient Heat Flow 
# Through Multilayered Slabs,” International Journal of Heat and Mass Transfer 26(11):1685-1693, 1983

def search(Δθmin,R,C):
    # %% Import required libraries
    import numpy as np
    from scipy.optimize import brentq as fzero
    import matplotlib.pyplot as plt # not needed unless plotting options are enabled
    from sourceCode.TransferMatrix import Matrix, dMatrix
    # %% Special numbers and tolerances
    eps=np.finfo(np.float64).eps # Smallest positive floating point in numpy
    rootTol=eps # Root finding technique tolerance
    # %% Plotting options. Attention! Should be deactivated when code runs in GT-Power - otherwise GT crashes.
    plotOn=0 # 0/1 - off/on Print plot window with roots of function D (and C)
    plotSteadyStateOn=0 # 0/1 - off/on Print plot to check if the steady state response is between the limits.
    if plotOn==1: plt.figure()

    # %% Tuning parameters for numerical accuracy
    βmax=23/Δθmin # e^(-βmax*Δθmin)>ε | βmax*Δθmin>23 for accuracy 10^(-10)~e^(-23) or 10 for accuracy 5*10^(-5)~e^(-10), see ref. [1]
    numRootsMax=1e3 # [-] Maximum number of roots as an upper limit to prevent code crashing
    SteadyStateTol=1e-1 #
    # %% Criterias to define when root search should terminate
    def criteria(numRootsMax,Rtot,βm,XsteadyState):
        numCriterias=2
        flags=np.zeros((numCriterias,1), dtype=bool) # Individual flags for each condition
        if len(βm)+1>numRootsMax:
            flags[0]=True
        if len(βm)>0 and numCriterias==2: # At least one root has been captured
            if βm[-1]>βmax and np.sum(XsteadyState)>(1-SteadyStateTol)*Rtot:
                flags[1]=True
        criteriaFlag=True if np.any(flags)==True else False
        return criteriaFlag
    
    def getXsteadyState(βm,Bm,dDm):
        return Bm/(βm*dDm)
    
    Rtot=np.sum(R) # [m**2-K/W] Total thermal resistance of the multi-layer
    N=0 # Counter of C sign changes since a D root was found
    β=eps # Initial evaluation
    pastSignD=np.sign(Matrix(β,R,C)[1,1]) # sign of D[eps]
    pastSignC=np.sign(Matrix(β,R,C)[1,0]) # sign of C[eps]
    pastRoot=[] # Dummy variable to track last root captured

    βm=[] # Poles/Roots of D(s)
    Bm=[] # Coefficient B(s) of overall transfer matrix evaluated at βm
    dDm=[] # Coefficient D(s) of overall diff transfer matrix evaluated at βm
    XsteadyState=[] # Response factor at steady-state limiting case
    dβ=0.5 # Step size along β-axis
    while not criteria(numRootsMax,Rtot,βm,XsteadyState):
        
        # %% Evaluate A,B,C,D terms
        [[AA,BB],
         [CC,DD]]=Matrix(β,R,C)
        
        # %% Plotting options
        if plotOn==1:
            plt.plot(np.sqrt(β),DD,'o')
            plt.plot(np.sqrt(β),CC/200000,'+')
        # %% Check if signs of current and last value of D have been altered
        if np.sign(DD)!=pastSignD:
            # Root of D is found in this interval
            root=fzero(lambda β: Matrix(β,R,C)[1,1], a=β-dβ+eps, b=β, xtol=rootTol) # Find root of D @ [β-dβ+eps,β]
            if N==1: N=0 # reset the counter
            pastSignC=np.sign(Matrix(root,R,C)[1,0]) # record sign of C @ root of D(β)
            # Record βm,Bm,dDm terms that are going to be used to compute the X/Y response function
            βm=np.append(βm,root)
            Bm=np.append(Bm,Matrix(root,R,C)[0,1])
            dDm=np.append(dDm,dMatrix(root,R,C)[1,1])
            XsteadyState=np.append(XsteadyState,getXsteadyState(βm[-1],Bm[-1],dDm[-1]))
            # Plotting options
            if plotOn==1: plt.plot(np.sqrt(root),Matrix(root,R,C)[1,1],'*r',markersize=16,markeredgecolor='k')
            
        # %% Check if signs of current and last value of C have been altered
        if np.sign(CC)!=pastSignC and pastSignC!=0: 
            # For the second case, the 0 sign occurs when C is evaluated @ s=eps. 
            # Need to avoid the case the if-statement will be true then
            if len(βm)>0:
                if pastRoot==βm[-1]: N+=1
            elif len(βm)==0: # still βm=[]
                if np.sign(DD)==pastSignD and (N==0 or N==1): N=2 
                
        # %% Set the current values to the previous for the next iteration
        pastSignD=np.sign(DD); pastSignC=np.sign(CC)
        
        # %% If the following is true, two roots of function D have been missed, without evaluating them
        
        if N==2: # two roots of D have been jumped over        
            # Root of D is found in this interval
            N=0 # reset counter
            rootOfC   =fzero(lambda β: Matrix(β,R,C)[1,0], a=β-dβ,   b=β      ,xtol=rootTol) # Find root of C @ [β-dβ   ,β      ]
            rootLeft  =fzero(lambda β: Matrix(β,R,C)[1,1], a=β-dβ,   b=rootOfC,xtol=rootTol) # Find root of D @ [β-dβ   ,rootOfC]
            rootRight =fzero(lambda β: Matrix(β,R,C)[1,1], a=rootOfC,b=β      ,xtol=rootTol) # Find root of D @ [rootOfC,β      ]

            pastSignC=np.sign(Matrix(rootRight,R,C)[1,0]) # record sign of C at the right-most root of D(β)
            # Record βm,Bm,dDm terms that are going to be used to compute the X/Y response function
            βm=np.append(βm,[rootLeft,rootRight])
            Bm=np.append(Bm,[Matrix(rootLeft,R,C)[0,1], Matrix(rootRight,R,C)[0,1]])
            dDm=np.append(dDm,[dMatrix(rootLeft,R,C)[1,1],dMatrix(rootRight,R,C)[1,1]])
            XsteadyState=np.append(XsteadyState,[getXsteadyState(βm[-2],Bm[-2],dDm[-2]) , 
                                                 getXsteadyState(βm[-1],Bm[-1],dDm[-1])])
            # Plotting option
            if plotOn==1:
                plt.plot(np.sqrt(rootLeft),  Matrix(rootLeft  ,R,C)[1,1],'sg',markersize=20,markeredgecolor='k')
                plt.plot(np.sqrt(rootRight),Matrix(rootRight,R,C)[1,1],'^g',markersize=20,markeredgecolor='k')
                plt.plot(np.sqrt(β),0,'x',markersize=12,markeredgecolor='k')
                
            β-=dβ # It begins the root search at the last evaluated β
            
        # Plotting options
        if plotOn==1: plt.plot(np.sqrt(β),N,'xk')
        
    # %% Store last root and calculate next Laplace variable step
        if len(βm)>0: pastRoot=βm[-1]
        β+=dβ
    # %% Plotting options to print windows
    if plotOn==1:
        plt.axhline(0, linestyle='--', color='k') # Horizontal zero grid line
        plt.show()
    if plotSteadyStateOn==1:
        plt.figure()
        plt.plot(βm,Rtot*np.ones(len(βm)),'-b') # Upper bound
        plt.plot(βm,np.zeros(len(βm)),'b') # Lower bound
        plt.plot(βm,XsteadyState,'-ok',label='Response')
        plt.plot(βm,np.cumsum(XsteadyState),'-or',label='Cumulative Response')
        plt.text(βm[-1], Rtot, 'Rtotal', fontsize=12)
        
        plt.plot(βm,(1-SteadyStateTol)*Rtot*np.ones(len(βm)),'-g')
        plt.xlabel(r'Poles')
        plt.ylabel(r'Resistance [K-m$^{2}$/W]')
        plt.title(r'Response at Steady State')
        plt.legend(); plt.show()
        
    # %% Output termns
    return βm,Bm,dDm
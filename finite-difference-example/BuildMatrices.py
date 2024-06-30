# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse

def BuildMatrices(layer,k,rhoc,NpL,L,h,N,dx,A_x,A_s):
    ''' Build C matrix'''
    d=np.zeros(N+1)
    for i in range(layer):
        d[np.sum(NpL[0:i]):np.sum(NpL[0:i+1])]=rhoc[i]*dx[i]*A_x
    # Correct end nodes
    d[0]=0.5*d[0]
    d[N]=0.5*rhoc[layer-1]*dx[layer-1]*A_x
    # Correct boundary nodes
    for i in range(layer-1):
        d[np.sum(NpL[0:i+1])]=0.5*rhoc[i]*dx[i]*A_x + 0.5*rhoc[i+1]*dx[i+1]*A_x
    # Load array d into a diagonal sparse matrix
    #C=sparse.eye(N+1)
    C=sparse.lil_matrix((N+1,N+1))
    C.setdiag(d)
    
    ''' Build K matrix'''
    K=sparse.lil_matrix((N+1,N+1),dtype=float)
    for i in range(layer):
        # make off-diagonal matrix of conductances
        updi=np.ones(NpL[i]-1);  lodi=np.ones(NpL[i]-1)
        updi=-k[i]*A_x/dx[i]*updi
        lodi=-k[i]*A_x/dx[i]*lodi
        # put diagnonal elements into K matrix at the correct row range for layer
        K[np.sum(NpL[0:i]):np.sum(NpL[0:i+1]),np.sum(NpL[0:i]):np.sum(NpL[0:i+1])]=sparse.diags([lodi,updi],[-1,1])
        # fix boundary nodes
        K[np.sum(NpL[0:i+1])-1,np.sum(NpL[0:i+1])]=-k[i]*A_x/dx[i]
        K[np.sum(NpL[0:i+1]),np.sum(NpL[0:i+1])-1]=-k[i]*A_x/dx[i]
    # sum to find diagonal elements
    d=np.array(-np.sum(K,axis=0))
    K=K + sparse.diags(d,[0])
    
    ''' Build H matrix -     Convective boundary at end node only'''
    H=sparse.lil_matrix((N+1,N+1),dtype=float)
    H[N,N]=h*A_s;
    
    S=K+H
    return C,S,H
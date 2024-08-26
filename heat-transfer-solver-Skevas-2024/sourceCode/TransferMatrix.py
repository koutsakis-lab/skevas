# -*- coding: utf-8 -*-
import numpy as np
eps=np.finfo(np.float64).eps

def dMatrix(β,R,C):
    def dMatrixPerLayer(β,R,C): # input of R and C are values
        dMatrixPerLayer=np.zeros((2,2))
        if β!=eps: # β can be anything but the limit where β->0 (eps)
            X=np.sqrt(β*R*C)    
            dMatrixPerLayer[0,0]= np.sin(X)/X
            dMatrixPerLayer[0,1]= 1/(β*C)*(np.sin(X)/X-np.cos(X))
            dMatrixPerLayer[1,0]=   (1/R)*(np.sin(X)/X+np.cos(X))
            dMatrixPerLayer[1,1]= dMatrixPerLayer[0,0]
        elif β==eps: # β->0, limits derived by hand
            dMatrixPerLayer[0,0]= 1
            dMatrixPerLayer[0,1]= R/3
            dMatrixPerLayer[1,0]= 2/R
            dMatrixPerLayer[1,1]= 1
        return (R*C/2)*dMatrixPerLayer
    numLayers=R.size
    dMatrix=np.zeros((2,2))
    for i in range(0,numLayers):
        iΠMatrices=dMatrixPerLayer(β,R[i],C[i])
        if numLayers>1:
            if i>0:
                ΠMatrixLeft=Matrix(β,R[0:i],C[0:i])
                iΠMatrices=np.matmul(ΠMatrixLeft, iΠMatrices)
            if i<numLayers-1:
                ΠMatrixRight=Matrix(β,R[i+1:numLayers],C[i+1:numLayers])
                iΠMatrices=np.matmul(iΠMatrices, ΠMatrixRight)
        dMatrix=np.add(dMatrix, iΠMatrices)
    return dMatrix

def Matrix(β,R,C): # input of R and C are arrays
    def MatrixPerLayer(β,R,C): # input of R and C are values
        MatrixPerLayer=np.zeros((2,2))
        if β!=eps: # β can be anything but the limit where β->0 (eps)
            if C==eps: # Negligible specific heat case
                MatrixPerLayer[0,0]= 1
                MatrixPerLayer[0,1]= R
                MatrixPerLayer[1,0]= 0
                MatrixPerLayer[1,1]= 1
            else: # normal layer
                X=np.sqrt(β*R*C)
                MatrixPerLayer[0,0]= np.cos(X)
                MatrixPerLayer[0,1]= (R/X)*np.sin(X)
                MatrixPerLayer[1,0]=-(X/R)*np.sin(X)
                MatrixPerLayer[1,1]= MatrixPerLayer[0,0]
        elif β==eps: # β->0, limits derived by hand
            MatrixPerLayer[0,0]= 1
            MatrixPerLayer[0,1]= R
            MatrixPerLayer[1,0]= 0
            MatrixPerLayer[1,1]= 1
        return MatrixPerLayer
    numLayers=R.size
    Matrix=MatrixPerLayer(β,R[0],C[0])
    for i in range(1,numLayers):
        Matrix=np.matmul(Matrix,MatrixPerLayer(β,R[i],C[i])) # 2x2 by 2x2 matrix product
    return Matrix
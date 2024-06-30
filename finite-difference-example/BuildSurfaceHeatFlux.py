# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def HeatFluxArtificial(t, f, q_prime, A_s, profile):
    #can take either a single value of t or an array
    q = 0 # if wrong string is provided 
    if profile=='step change':
        q = q_prime*A_s*np.ones(np.size(t))    # Uniform heat flux
        
    elif profile=='gauss':
        # Gaussian distribution
        t_o=np.pi/2 # Location of peak heat flux (absolute)
        wid=np.pi/16 # Pulse width
        
        q = 0
        for i in range(f):
            a=(2*np.pi*f*t-(2*np.pi*i+t_o))/wid
            q = q + q_prime*A_s*np.exp(-0.5*a*a)
    
    elif profile == 'sin':
        q = q_prime*A_s*np.power(np.sin(2*np.pi*f*t),2)    # Sinusoidal variation
        # q = q_prime*A_s*np.power(np.sin(2*np.pi*f*t),1)    # Sinusoidal variation
        
    return q
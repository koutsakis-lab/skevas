# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def HeatFlux(t, f, q_prime, A_s, profile):
    #can take either a single value of t or an array

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
            
    elif profile=='gauss with sin':
        # Gaussian distribution
        t_o=np.pi/1.8 # Location of peak heat flux (absolute)
        wid=np.pi/32 # Pulse width
        q_background=0.05e+6 # [W/m2] Background heat flux magnitude
        q = q_background*A_s*np.sin(2*np.pi*f*t)
        for i in range(f):
            a=(2*np.pi*f*t-(2*np.pi*i+t_o))/wid
            q = q + q_prime*A_s*np.exp(-0.5*a*a)
            
    elif profile=='triangular':
        
        def exactTriangleHeatFlux(t_plus, q_nominal):
            q_plus = np.zeros_like(t_plus)
            
            for i in range(len(q_plus)):
            
                if t_plus[i] <= 0.1:
                    q_plus[i] = 0
                elif t_plus[i] > 0.1 and t_plus[i] <= 0.5:
                    q_plus[i] = ( 2.5*t_plus[i] - 0.25 )*q_nominal
                elif t_plus[i] > 0.5 and t_plus[i] <= 0.9:
                    q_plus[i] = (-2.5*t_plus[i] + 9/4)*q_nominal
                else:
                    q_plus[i] = 0
    
            # plt.figure(1)
            # plt.plot(t_plus,q_plus, label='exact')
            # plt.ylabel('Heat Flux [-]')
            # plt.xlabel('Time [-]')
            return q_plus
        
        q = exactTriangleHeatFlux(t, q_prime)
        # plt.plot(t,q)
    elif profile=='ramp':
        q = t * q_prime
    
    elif profile=='negative ramp':
        q = (1-t) * q_prime\
    
    elif profile == 'sin':
        q = q_prime*A_s*np.power(np.sin(2*np.pi*f*t),2)    # Sinusoidal variation
        # q = q_prime*A_s*np.power(np.sin(2*np.pi*f*t),1)    # Sinusoidal variation
        
    return q
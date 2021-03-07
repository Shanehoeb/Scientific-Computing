# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:05:49 2021

@author: Shane
"""
import numpy as np
from solve_to import solve_to


def solve_ode(x_0,t_0,t_array,stepsize, deltat_max):
    #generates a series of numerical solution estimates x_1, x_2, x_3...
    # Input : x_0,t_0 initial conditions, array of times at which to approximate t_array,
    #         stepsize of euler step and maximum time step for approximations
    results = []
    t_array = sorted(t_array,key=float)
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max)
        x_0,t_0 = x_value
        results=np.append(results,x_value[0])
    return t_array,results
    
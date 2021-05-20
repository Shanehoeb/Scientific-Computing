import numpy as np
def rk4_step(stepsize, func, X, t):
     #Performs a single 4th order Runge-Kutta step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize,func the ODE
    # Output : Updated pair of values after Runge-Kutta step of stepsize
    k_1 = func(t, X)
    k_2 = func(t + stepsize * 1/2, X + stepsize*k_1*1/2)
    k_3 = func(t + stepsize * 1/2, X + stepsize*k_2*1/2)
    k_4 = func(t + stepsize, X + stepsize*k_3)
    return X + 1/6*stepsize*(k_1 + 2*k_2 + 2*k_3 + k_4), t + stepsize

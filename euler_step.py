import numpy as np
import matplotlib.pyplot as plt


def euler_step(stepsize,x_n,t_n):
    return x_n + stepsize*x_n, t_n + stepsize

def solve_to(x_1,t_1,t_2,stepsize):
    while t_1 < t_2:
        delta_t = t_2 - t_1
        if delta_t < stepsize:
            stepsize = delta_t
        x_1,t_1 = euler_step(stepsize, x_1, t_1)  
    return x_1, t_1


stepsize = 0.125
x_0 = 1
t_0 = 0

a = solve_to(x_0,t_0,1,stepsize)
print(a)

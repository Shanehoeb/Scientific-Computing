import numpy as np
import matplotlib.pyplot as plt



def euler_step(stepsize,current_approx,x_n,t_n):
    return x_n + stepsize*current_approx, t_n + stepsize

def expression(x,t):
    return x
stepsize = 0.125
x_0 = 1
t_0 = 0

a = euler_step(stepsize,expression(x_0,t_0),x_0,t_0)
print(a)
import numpy as np
import matplotlib.pyplot as plt
from solve_ode import solve_ode

#Params
stepsize_list = []
step = 0
for i in range(0,10000):
    step += 1/10000
    stepsize_list.append(step)
    
x_0 = 1
t_0 = 0
t_array = np.random.rand(10)
t_array = np.append(t_array,1)
deltat_max = 2

error_list = []

def func(x,t):
    return x

if __name__ == '__main__':

    for stepsize in stepsize_list:
        sol = solve_ode(x_0,t_0,t_array,stepsize,deltat_max,func)
        error = np.absolute(np.exp(1)-sol[1][-1])
        error_list.append(error)

    fig = plt.figure()
    ax = plt.gca()
    plt.scatter(stepsize_list,error_list,1,c="r")
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()

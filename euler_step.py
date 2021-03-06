import numpy as np
import matplotlib.pyplot as plt

def euler_step(stepsize,x_n,t_n):
    #Performs a single euler step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize
    # Output : Updated pair of values after euler step of stepsize
    return x_n + stepsize*x_n, t_n + stepsize

def solve_to(x_1,t_1,t_2,stepsize,deltat_max):
    #solves from x_1,t_1 to x_2,t_2 in steps no bigger than deltat_max
    # Input : x_1,t_1 pair of coordinates, t_2 time at which we want to approximate,
    #         stepsize of the euler step and the maximum difference between 
    #         input and output time deltat_max
    if t_2 - t_1 < deltat_max:
        while t_1 < t_2:
            delta_t = t_2 - t_1
            if delta_t < stepsize:
                stepsize = delta_t
            x_1,t_1 = euler_step(stepsize, x_1, t_1)  
        return x_1, t_1
    else:
        print("Difference is too big ! Delta t max exceeded.")
        
        
def solve_ode(x_0,t_0,t_array,stepsize, deltat_max):
    #generates a series of numerical solution estimates x_1, x_2, x_3...
    # Input : x_0,t_0 initial conditions, array of times at which to approximate t_array,
    #         stepsize of euler step and maximum time step for approximations
    results = []
    error = 0
    t_array = sorted(t_array,key=float)
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max)
        x_0,t_0 = x_value
        results=np.append(results,x_value[0])
    return t_array,results
    

stepsize_list = []
step = 0
for i in range(0,10000):
    step += 1/10000
    stepsize_list.append(step)
    
x_0 = 1
t_0 = 0
t_array = np.random.rand(10)
t_array=np.append(t_array,1)
deltat_max = 2

error_list = []

for stepsize in stepsize_list:
    sol = solve_ode(x_0,t_0,t_array,stepsize,deltat_max)
    error = np.absolute(np.exp(1)-sol[1][-1])
    error_list.append(error)

fig = plt.figure()
ax = plt.gca()
plt.scatter(stepsize_list,error_list,1,c="r")
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
    

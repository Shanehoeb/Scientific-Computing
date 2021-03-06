import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def euler_step(stepsize,x_n,t_n):
    #Performs a single euler step for a value of x, t and a stepsize
    return x_n + stepsize*x_n, t_n + stepsize

def solve_to(x_1,t_1,t_2,stepsize,deltat_max):
    #solves from x_1,t_1 to x_2,t_2 in steps no bigger than deltat_max
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
    results = []
    error = 0
    t_array = sorted(t_array,key=float)
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max)
        x_0 = timestamp
        error += np.absolute(np.exp(timestamp) - x_value[0])
        results.append(x_value[0])
    return t_array,results,error
    


stepsize = 0.2
x_0 = 1
t_0 = 0
t_array = np.random.rand(1000)
deltat_max = 2

a = solve_ode(x_0,t_0,t_array,stepsize,deltat_max)
print(a[1])

plt.scatter(a[0],a[1],c="r")
plt.show()

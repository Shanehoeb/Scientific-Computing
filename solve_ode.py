import numpy as np
from solve_to import solve_to


def solve_ode(func, t_span, x_0, stepsize, method, deltat_max):
    #generates a series of numerical solution estimates x_1, x_2, x_3...
    # Input
    # -func : function to solve
    # -t_span : tuple for range of time to solve for
    # -X_0 : initial guess
    # -stepsize : stepsize for solving
    # -method : 4th order Runge-Kutta or Euler
    # deltat_max : maximum step in time
    x_list = []
    y_list = []
    t_array = np.linspace(t_span[0], t_span[1], 11)
    t_0 = t_span[0]
    #
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max, func, method)
        x_0, t_0 = x_value
        x_list.append(x_value[0][0])
        y_list.append(x_value[0][1])
        if timestamp == 1:
            #print(x_value[0][0],x_value[0][1])
            pass
    return t_array, x_list, y_list


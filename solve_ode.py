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
    results = []
    t_array = np.linspace(t_span[0],t_span[1],201)
    t_0 = t_span[0]
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max, func, method)
        x_0, t_0 = x_value
        results = np.append(results, x_value[0])
    return t_array, results

#def solve_ode(func,t_span,X_0, stepsize,method, deltat_max):
    #generates a series of numerical solution estimates x_1, x_2, x_3...
    # Input : x_0,t_0 initial conditions, array of times at which to approximate t_array,
    #         stepsize of euler step and maximum time step for approximations
 #   results = []
  #  t_array = np.linspace(t_span[0], t_span[1], 201)
   # t_0 = t_span[0]
    #for timestamp in t_array:
     #   x_value = solve_to(X_0, t_0, timestamp, stepsize, deltat_max, func, method)
      #  x_0, t_0 = x_value
       # results = np.append(results, x_value[0])
    #return t_array, results


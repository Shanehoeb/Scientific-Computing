import numpy as np


def rk4_step(stepsize, func, X, t):
    # Performs a single 4th order Runge-Kutta step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize,func the ODE
    # Output : Updated pair of values after Runge-Kutta step of stepsize
    X = np.array(X)
    k_1 = func(t, X)
    k_2 = func(t + stepsize * 1 / 2, X + stepsize * k_1 * 1 / 2)
    k_3 = func(t + stepsize * 1 / 2, X + stepsize * k_2 * 1 / 2)
    k_4 = func(t + stepsize, X + stepsize * k_3)
    return X + 1 / 6 * stepsize * (k_1 + 2 * k_2 + 2 * k_3 + k_4), t + stepsize


def euler_step(stepsize, func, X, t):
    X = np.array(X)
    #Performs a single euler step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize,func the ODE
    # Output : Updated pair of values after euler step of stepsize
    return X + stepsize*func(t, X), t + stepsize


def solve_to(X,t_1,t_2,stepsize,deltat_max,func,method):
    #solves from x_1,t_1 to x_2,t_2 in steps no bigger than deltat_max
    # Input : x_1,t_1 pair of coordinates, t_2 time at which we want to approximate,
    #         stepsize of the euler step and the maximum difference between
    #         input and output time deltat_max
    if t_2 - t_1 < deltat_max:
        while t_1 < t_2:
            delta_t = t_2 - t_1
            if delta_t < stepsize:
                stepsize = delta_t
            if method == 'euler':
                X = euler_step(stepsize, func, X, t_1)[0]
            if method == 'rk4':
                X = rk4_step(stepsize, func, X, t_1)[0]

            t_1 += stepsize
        return X, t_1
    else:
        print("Difference is too big ! Delta t max exceeded.")


def solve_ode(func, t_span, x_0, stepsize, method, deltat_max=2):
    #generates a series of numerical solution estimates x_1, x_2, x_3...
    # Input
    # -func : function to solve
    # -t_span : tuple for range of time to solve for
    # -X_0 : initial conditions
    # -stepsize : stepsize for solving
    # -method : 4th order Runge-Kutta or Euler
    # deltat_max : maximum step in time
    x_list = []
    solve_points = 201
    if t_span[1] > 200:
        solve_points = round(t_span[1]+1)
    t_array = np.linspace(t_span[0], t_span[1], solve_points)
    t_0 = t_span[0]
    #
    for timestamp in t_array:
        x_value = solve_to(x_0, t_0, timestamp, stepsize, deltat_max, func, method)
        x_0, t_0 = x_value
        x_list.append(x_value[0])
    return t_array, x_list
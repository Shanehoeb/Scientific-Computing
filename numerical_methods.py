import numpy as np


def rk4_step(stepsize, func, X, t):
    """Performs a single 4th order Runge-Kutta step.

    Parameters
    ---------
    stepsize : float
               Determines the size of the time step taken to compute the 4th
               order Runge-Kutta step.

    func : callable
           Callable function of the ODE to solve, in which the ODE is converted to a
           system of first order differential equations.

    X : array-like, size(D,)
        Array of length D the number of dimensions of the system containing
        the initial values at time t for all equations in the system.

    t : float
        Initial time stamp t, corresponding to the time for values in X.

    Returns
    ---------
    Tuple of length 2 of values after single 4th order Runge-Kutta step

    Tuple[0] : array-like, size(D,)
               Updated array of values after the 4th order Runge-Kutta step.
               Same structure as X, entries updated after calculations.

    Tuple[1] : float
               Time stamp of the updated values, equal to the sum of initial
               time t and the step size stepsize.
    """
    # Convert tuple of initial conditions to Numpy array
    X = np.array(X, dtype='float64')
    # Compute necessary values for calculations
    k_1 = func(t, X)
    k_2 = func(t + stepsize * 1 / 2, X + stepsize * k_1 * 1 / 2)
    k_3 = func(t + stepsize * 1 / 2, X + stepsize * k_2 * 1 / 2)
    k_4 = func(t + stepsize, X + stepsize * k_3)
    # Return updated values after rk4 step
    return X + 1 / 6 * stepsize * (k_1 + 2 * k_2 + 2 * k_3 + k_4), t + stepsize


def euler_step(stepsize, func, X, t):
    """Performs a single Euler step.

    Parameters
    ---------
    stepsize : float
               Determines the size of the time step taken to compute the Euler step.

    func : callable
           Callable function of the ODE to solve, in which the ODE is converted to a
           system of first order differential equations.

    X : array-like, size(D,)
        Array of length D the number of dimensions of the system containing
        the initial values at time t for all equations in the system.

    t : float
        Initial time stamp t, corresponding to the time for values in X.

    Returns
    ---------
    Tuple of length 2 of values after single Euler step

    Tuple[0] : array-like, size(D,)
               Updated array of values after the Euler step. Same structure
               as X, entries updated after calculations.

    Tuple[1] : float
               Time stamp of the updated values, equal to the sum of initial
               time t and the step size stepsize.
    """
    # Convert tuple of initial conditions to Numpy array
    X = np.array(X, dtype='float64')
    # Return updated values after Euler step
    return X + stepsize*func(t, X), t + stepsize


def solve_to(X, t_1, t_2, stepsize, deltat_max, func, method):
    """ Solves differential equations for specified time period, and returns
        values for the limits of the time period.

    Parameters
    ---------
    X : array-like, size(D,)
        Array of length D the number of dimensions of the system containing
        the initial values at time t_1 for all equations in the system.

    t_1 : float
          Initial time stamp t, corresponding to the time for values in X.

    t_2 : float
          Time at which the user wants the ODEs' numerical solution.

    stepsize : float
               Determines the size of the time step taken to compute the step.

    deltat_max : float
                 Maximum time difference between t_1 and t_2.

    func : callable
           Callable function of the ODE to solve, in which the ODE is converted to a
           system of first order differential equations.

    method : string
             Desired method to solve the ODE. Options are 4th order Runge-Kutta
             or Euler. These must be passed as "rk4" or "euler" respectively.

    Returns
    ---------
    Tuple of length 2 sontaining a Numpy array and a float.

    Tuple[0] : Numpy array of numerical solutions of the ODE at time t_2. Same
               structure as X, entries updated after iterating calculations of
               chosen method andincrementing time by the stepsize.

    Tuple[1] : time t_2 (float), for convenience in further calculations.
    """
    # Check if step is smaller than maximum step
    if t_2 - t_1 < deltat_max:
        # While the end time limit is not reached
        while t_1 < t_2:
            # Check if time left to calculate for is smaller than step size
            delta_t = t_2 - t_1
            # If it is, use this smaller time step as our step size for final step
            if delta_t < stepsize:
                stepsize = delta_t
            # If Euler is chosen, compute Euler
            if method == 'euler':
                X = euler_step(stepsize, func, X, t_1)[0]
            # If rk4 is chosen, compute rk4
            if method == 'rk4':
                X = rk4_step(stepsize, func, X, t_1)[0]
            #Increment time
            t_1 += stepsize
        return X, t_1
    else:
        # If max time difference exceeded
        print("Difference is too big ! Delta t max exceeded.")


def solve_ode(func, t_span, X, stepsize, method, deltat_max=2, points=0):
    """Function that solve an initial value problem for a system of ODEs.

    Parameters
    ---------
    func : callable
           Callable function of the ODE to solve, in which the ODE is converted to a
           system of first order differential equations.

    t_span : tuple of floats, size(2,)
             Tuple containing time limit values defining a time period for which
             to solve for.

    X : array-like, size(D,)
        Array of length D the number of dimensions of the system containing
        the initial values at time t_ span[0] for all equations in the system.

    stepsize : float
               Determines the size of the time step taken to compute the step.

    deltat_max : float
                 Maximum time difference between values in t_span.

    method : string
             Desired method to solve the ODE. Options are 4th order Runge-Kutta
             or Euler. These must be passed as "rk4" or "euler" respectively.
    points : int
             Number of time stamps to calculate the ODE solution for between
             values of t_span. Default is 201 if t_span period is inferior to 200,
             Otherwise, equal to t_span period + 1 (rounded).
    Returns
    ---------
    Tuple of length 2 containing two Numpy arrays.

    t_array : array of equally spaced time stamps between the two time limits in
              t_span. Length of this array depends depends on points parameter.

    solution : array of numerical solutions of the system of ODEs corresponding to
               the time stamps in t_array. Length is same as t_array.
    """
    #If no points argument is passed
    if points == 0:
        # If time period superior to 200
        if t_span[1]-t_span[0] > 200:
            # Number of points equal to time period + 1 (round)
            solve_points = round(t_span[1]+1)
        else:
            # Otherwise, default value 201
            solve_points = 201

        # Array of equally spaced time stamps in time period
        t_array = np.linspace(t_span[0], t_span[1], solve_points)
    else:
        t_array = np.linspace(t_span[0], t_span[1], points)
    t_0 = t_span[0]
    x_list = []
    # Iterate over time stamp array
    for timestamp in t_array:
        # Get value from last timestamp to new one
        x_value = solve_to(X, t_0, timestamp, stepsize, deltat_max, func, method)
        X, t_0 = x_value
        # Append solution for time stamp
        x_list.append(X)
    return t_array, x_list
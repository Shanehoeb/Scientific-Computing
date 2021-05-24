from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve
from math import nan
import matplotlib.pyplot as plt
from numerical_methods import*
import numpy as np
import sympy as sm


def scipy_solver(ode, initialu, t_span, plot=False):
    """ Uses Scipy' solve_ivp to solve initial value problem for an ODE.

    Parameters
    ---------
    ode : callable
          Callable function of the ODE to solve, in which the ODE is converted to a
          system of first order differential equations.

    initialu: Array-like of floats, size(D,)
              Array of length D the number of dimensions of the system containing
              the initial values at time t for all equations in the system.

    t_span : tuple of floats, size(2,)
             Tuple containing time limit values defining a time period for which
             to solve for.
    plot: bool
          Boolean value passed by the user; if True, a plot of the calculated
          solutions of the ODE over the time period will be generated

    Returns
    ---------
    Tuple of length 2 containing two Numpy arrays.

    sol.t : array of equally spaced time stamps between the two time limits in
            t_span. Length of this array depends depends on points parameter.

    sol.y : array of numerical solutions of the system of ODEs corresponding to
            the time stamps in t_array. Length is same as t_array.

    """
    # Solve using scipy
    sol = solve_ivp(ode, t_span, initialu)
    # If plot argument is true, plot time simulation of the ODE
    if plot:
        for i in range(len(sol.y)):
            plt.plot(sol.t, sol.y[i, :])
        plt.title("Time Series Simulation for initial conditions %s" % str(initialu))
        plt.show()
    # Return solution
    return sol.t, sol.y


def my_solver(ode, initialu, t_span, stepsize=0.005, method="rk4", deltat_max=2, plot=True):
    """ Uses custom solver to solve initial value problem for an ODE.

    Parameters
    ---------
    ode : callable
          Callable function of the ODE to solve, in which the ODE is converted to a
          system of first order differential equations.

    initialu: Array-like of floats, size(D,)
              Array of length D the number of dimensions of the system containing
              the initial values at time t for all equations in the system.

    t_span : tuple of floats, size(2,)
             Tuple containing time limit values defining a time period for which
             to solve for.

    stepsize : float
               Determines the size of the time step taken to compute the step for
               custom solver.

    method : string
             Desired method to solve the ODE. Options are 4th order Runge-Kutta
             or Euler. These must be passed as "rk4" or "euler" respectively.

    deltat_max : float
                 Maximum time difference between two timestamps for solving.

    plot: bool
          Boolean value passed by the user; if True, a plot of the calculated
          solutions of the ODE over the time period will be generated.

    Returns
    ---------
    Tuple of length 2 containing two Numpy arrays.

    t_array : array of equally spaced time stamps between the two time limits in
              t_span. Length of this array depends depends on points parameter.

    solution : array of numerical solutions of the system of ODEs corresponding to
               the time stamps in t_array. Length is same as t_array.
        """
    # Solve ODE using custom solver
    t_array, sol = solve_ode(ode, t_span, initialu, stepsize, method, deltat_max)
    sol = np.array(sol)
    solution = []
    # Reformat to match scipy versiom
    for i in range(len(sol[0])):
        solution.append(sol[:, i])
    # Plot Optional
        if plot:
            plt.plot(t_array, sol[:, i])
    if plot:
        plt.title("Time Series Simulation for initial conditions %s" % (initialu,))
        plt.show()

    return t_array, solution


def time_simulation(ode, initialu, t_span, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, plot=False):
    """ Uses Scipy' solve_ivp or a custom solver to solve initial value problem
        for an ODE over a specified time period.

    Parameters
    ---------
    ode : callable
          Callable function of the ODE to solve, in which the ODE is converted to a
          system of first order differential equations.

    initialu: Array-like of floats, size(D,)
              Array of length D the number of dimensions of the system containing
              the initial values at time t for all equations in the system.

    t_span : tuple of floats, size(2,)
             Tuple containing time limit values defining a time period for which
             to solve for.


    solver : string, "scipy" or "custom"
             Choice of solver to use, either scipy's using solve_ivp or custom


    method : string
             Desired method to solve the ODE. Options are 4th order Runge-Kutta
             or Euler. These must be passed as "rk4" or "euler" respectively.
             Only for custom solver.

    stepsize : float
               Determines the size of the time step taken to compute the step for
               custom solver. Only for custom solver.

    deltat_max : float
                 Maximum time difference between two timestamps for solving.
                 Only for custom solver.

    plot: bool
          Boolean value passed by the user; if True, a plot of the calculated
          solutions of the ODE over the time period will be generated.

    Returns
    ---------
    Tuple of length 2 containing two Numpy arrays.

    t_array : array of equally spaced time stamps between the two time limits in
              t_span. Length of this array depends depends on points parameter.

    solution : array of numerical solutions of the system of ODEs corresponding to
               the time stamps in t_array. Length is same as t_array.
        """
    if solver == "scipy":
        return scipy_solver(ode, initialu, t_span, plot)
    elif solver == "custom":
        return my_solver(ode, initialu, t_span, stepsize, method, deltat_max, plot)


def orbit(ode, initialu, t_span, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, plot=False):
    """ Compute phase-portrait (orbits) of an initial value problem for an ODE.

        Parameters
        ---------
        ode : callable
              Callable function of the ODE to solve, in which the ODE is converted to a
              system of first order differential equations.

        initialu: Array-like of floats, size(D,)
                  Array of length D the number of dimensions of the system containing
                  the initial values at time t for all equations in the system.

        t_span : tuple of floats, size(2,)
                 Tuple containing time limit values defining a time period for which
                 to solve for.


        solver : string, "scipy" or "custom"
                 Choice of solver to use, either scipy's using solve_ivp or custom


        method : string
                 Desired method to solve the ODE. Options are 4th order Runge-Kutta
                 or Euler. These must be passed as "rk4" or "euler" respectively.
                 Only for custom solver.

        stepsize : float
                   Determines the size of the time step taken to compute the step for
                   custom solver. Only for custom solver.

        deltat_max : float
                     Maximum time difference between two timestamps for solving.
                     Only for custom solver.

        plot: bool
              Boolean value passed by the user; if True, a plot of the calculated
              solutions of the ODE over the time period will be generated.

        Returns
        ---------
        _ : array of equally spaced time stamps between the two time limits in
            t_span. Length of this array depends depends on points parameter.

        sol : array of numerical solutions of the system of ODEs corresponding to
              the time stamps in t_array. Length is same as t_array. Each entry of sol
              corresponds to the solutions of one of the ODEs in the system. sol[0]
              and sol[1] are used to plot orbit.
            """
    if solver == "scipy":
        _, sol = scipy_solver(ode, initialu, t_span, solver)
        if plot:
            # Plot orbit
            plt.plot(sol[0], sol[1], "b-")
            plt.title("Periodic Orbit")
            plt.show()
        return _, sol
    elif solver == "custom":
        _, sol = my_solver(ode, initialu, t_span, stepsize, method, deltat_max, plot=False)
        if plot:
            plt.plot(sol[0], sol[1], "b-")
            plt.title("Periodic Orbit  for initial conditions %s" % (initialu,))
            plt.show()
        return _, sol


def nullcline(ode, u0range, index=0, points=101):
    """ Returns the nullcline for one of the ODE variables for a specified range
        of values.

        Parameters
        ---------
        ode : callable
              Callable function of the ODE to solve, in which the ODE is converted to a
              system of first order differential equations.

        u0range: tuple of floats, size(D,)
                 Tuple containing the limit values of the range for which the
                 nullcline is computed.

        index: int
               Integer representing the variable of interest in the ODE function.

        points: int
                Number of points to compute the nullcline at. If not accurate enough,
                increase. Default is 101.

        Returns
        ---------
        Tuple of Numpy arrays containing the variable values at the nullcline of the
        ODE.

        """
    # Set one to 0, Calculate other
    Vval = np.linspace(min(u0range), max(u0range), points)
    Nval = np.zeros((len(Vval), len(u0range)))
    # Iterate and calculate nullcline
    for (i, V) in enumerate(Vval):
        result = root(lambda N: ode(nan, (V, N))[index], np.array([min(u0range), max(u0range)]))
        if result.success:
            Nval[i] = result.x
        else:
            Nval[i] = nan
    else:
        return (Vval, Nval)


def find_equilibria(ode, nb_dim):
    """Uses Sympy library to find all equilibria of a system

       Parameters
       ---------
        ode : callable
              Callable function of the ODE to solve, in which the ODE is converted to a
              system of first order differential equations.

        nb_dim : int
                 Number of dimensions of the ODE (number of first order equations
                  in system)

        Returns
        ---------
        Array of tuples containing equilibrium coordinates
     """
    symbols = []
    eqs = []
    # Create list of symbols
    for i in range(nb_dim):
        r = sm.symbols(str(i))
        symbols.append(r)
    # Compute ode for symbols
    u = ode(np.nan, symbols)
    # Solve symbolic expression
    for symbol in u:
        eqs.append(sm.Eq(symbol, 0))
    # Return equilibria
    equilibria = sm.solve(eqs, symbols)
    return equilibria

import phaseportraits as pp
import numpy as np
from scipy.optimize import fsolve


def phase_condition(ode, initialu, index=0):
    """Calculates the derivative at time t=0, which is an appropriate phase condition
       to then find an ODE's limit cycles.

       Parameters
       ---------
       ode : callable
             Callable function of the ODE to solve, in which the ODE is converted to a
             system of first order differential equations.

       initialu: Array-like of floats, size(D,)
                 Array of length D the number of dimensions of the system containing
                 the initial values at time t for all equations in the system.

       index : int
               Integer representing the variable of interest in the ODE function.

       Returns
       ---------

       Value of the derivative at time t = 0.
       """
    sol = np.array([ode(0, initialu)[index]])
    return sol


def vector_eq(init_guess, ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0):
    """Calculates phase condition & ODE integration condition.

       Parameters
       ---------
       init_guess : tuple of length 2
                    [0] Contains a Numpy array of initial values
                        of the ODE
                    [1] Contains a float of time t which is the guess
                        for the value of the period T of the ODE.
       ode : callable
             Callable function of the ODE to solve, in which the ODE is converted to a
             system of first order differential equations.
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
       Necessary equation to find limit cycle of ODEs. Numpy array at [0]
       Float at [1]
                    """
    initialu, t_period = init_guess[:-1], init_guess[-1]
    phase_cond = phase_condition(ode, initialu, index)
    t_array, solution = pp.time_simulation(ode, initialu, (0, t_period), solver, method, stepsize, deltat_max)
    values = []

    for i in range(len(solution)):
        values.append(solution[i][-1])
    return np.concatenate(((initialu - values), phase_cond))


def shoot(init_guess, ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0, plot=False):
    """Numerical shooting method to find an ODEs limit cycle/periodic orbit.

       Parameters
       ---------
       init_guess : tuple of length 2
                    [0] Contains a Numpy array of initial values
                        of the ODE
                    [1] Contains a float of time t which is the guess
                        for the value of the period T of the ODE.
       ode : callable
             Callable function of the ODE to solve, in which the ODE is converted to a
             system of first order differential equations.
    """
    cond = lambda guess, func: vector_eq(guess, func, solver=solver, method=method, stepsize=stepsize, deltat_max=deltat_max, index=index)
    solution = fsolve(cond, init_guess, ode)
    if plot:
        pp.orbit(ode, solution[:-1], (0, solution[-1]), plot=True)
    return solution

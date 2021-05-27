import matplotlib.pyplot as plt
from numerical_shooting import*


def natural_continuation(init_guess, ode, range, p_step, param_string, params, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0, plot=True):
    """Performs natural parameter continuation for an ODE and parameter of choice.

    Parameters
    ---------
    init_guess : tuple of length D+1 D dimensions of system
                    [0:D] Initial values of the ODE
                    [1] Contains a float of time t which is the guess
                        for the value of the period T of the ODE

    ode : callable
          Callable function of the ODE to solve, in which the ODE is converted to a
          system of first order differential equations.

    range : tuple, size(2,)
            tuple containing limit values of the varying parameter (floats)

    p_step : float
             Increment of parameter value at each step

    param_string : string
                   Key representing parameter value in parameter dictionary.

    params : dict
             Dictionary of the ODE parameters

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

    index: int
           Integer representing the variable of interest in the ODE function.

    plot: bool
          Boolean value passed by the user; if True, a plot of the calculated
          solutions of the ODE over the time period will be generated.

    Returns
    ---------
    Tuple of array of parameter values and values of a variable at that parameter value
    for limit cycle initial conditions found with numerical shooting.
    """
    points = round((range[1] - range[0])/p_step)
    # Array of parameter values
    param_list = np.linspace(range[0], range[1], points+1)
    values = []
    for element in param_list:
        # Increment parameter
        params[param_string] = element
        n_ode = lambda t, u: ode(t, u, params)
        try:
            # Solve
            sol = shoot(init_guess, n_ode, solver=solver, method=method, stepsize=stepsize, deltat_max=deltat_max, index=index, plot=False)
            values.append(sol[index])
            # New guess is last solution
            init_guess = sol
        except:
            print("Something went wrong at parameter value %s" % str(element))
            values.append(np.nan)
            # Errors sometimes encountered
    # Plot
    if plot:
        plt.plot(param_list, values, 'ro', label='num')
        plt.show()
    return param_list, values


# Failed attempts at pseudo arclength continuation
def pseudo_eq(sols):
    """

       Parameters
       ---------
       sols : tuple, size (2,)
              Tuple of numpy arrays containing solutions for (alpha, u) for two different
              alpha values where alpha is a parameter of the ODE
    """
    v_0, v_1 = sols
    diff = v_1 - v_0
    v_next = v_1 + diff
    eq = lambda v: np.dot((v-v_next), diff)
    return eq

def pseudo_arclength():
    pass
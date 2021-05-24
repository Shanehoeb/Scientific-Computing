import phaseportraits as pp

def phase_condition(ode, initialu, index):
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
       """
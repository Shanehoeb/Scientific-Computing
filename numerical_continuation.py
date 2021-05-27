import matplotlib.pyplot as plt
from numerical_shooting import*


def natural_continuation(init_guess, ode, range, p_step, param_string, params, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0, plot=True):
    points = round((range[1] - range[0])/p_step)
    param_list = np.linspace(range[0], range[1], points+1)
    values = []
    for element in param_list:
        params[param_string] = element
        n_ode = lambda t, u: ode(t, u, params)
        try:
            sol = shoot(init_guess, n_ode, solver=solver, method=method, stepsize=stepsize, deltat_max=deltat_max, index=index, plot=False)
            values.append(sol[index])
            init_guess = sol
        except:
            print("Something went wrong at parameter value %s" % str(element))
            values.append(np.nan)
    if plot:
        plt.plot(param_list, values, 'ro', label='num')
        plt.show()
    return param_list, values



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
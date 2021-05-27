import numpy as np
import ODEs as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numerical_continuation import*
import numerical_shooting as ns
import PDEs as pde
from math import pi


def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.11,
        "d": 0.1
    }
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))

#natural_continuation((0.79, 0.18, 30.), pred_prey, (0.1, 0.5), 0.005, "b", defaults_pred_prey(),solver="custom", method="rk4", stepsize=0.005, deltat_max=20, index=0, plot=True)

pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
a = ns.shoot((0.32, 0.32, 30.), pred_ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=20, index=0, plot=True)
print(a)
print(pp.time_simulation(pred_ode, a[:-1], (0, 100))[1][-1][-1])

def default_heat_params():
    return{
        "L": 1.,
        "kappa": 1.,
        "T": 0.5
    }

def u_I(x,params):
    # initial temperature distribution
    y = np.sin(pi*x/params["L"])
    return y

def h(x, t):
    return 2*x + t

#pde.pde_solver(u_I, params=default_heat_params(), mx=60, mt=8000, boundary_conds="zero", pde_type="heat_source", method="f-euler", heat_func=h, plot=True)

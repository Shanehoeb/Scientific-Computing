import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numerical_continuation import*
import numerical_shooting as ns
import PDEs as pde
from math import pi


def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.25,
        "d": 0.1
    }
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))

#natural_continuation((0.33, 0.33, 18), pred_prey, (0.1, 0.5), 0.001, "b", defaults_pred_prey(),solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0, plot=True)

#pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
#ns.shoot((0.33, 0.33, 18), pred_ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0, plot=False)
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
pde.pde_solver(u_I, params=default_heat_params(), method="f-euler", plot=True, mx=60, mt=8000)


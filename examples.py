import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numerical_methods as nm
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

pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())


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
pde.pde_solver(u_I, params=default_heat_params(), method="ck", plot=True)


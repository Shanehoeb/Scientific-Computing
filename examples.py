import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numerical_methods as nm
import numerical_shooting as ns

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

print(ns.phase_condition(pred_ode, (0.5,0.5), index=0))

ns.vector_eq((np.array((0.5,0.5)), 18), pred_ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, index=0)




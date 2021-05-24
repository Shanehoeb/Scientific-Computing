import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numerical_methods as nm


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






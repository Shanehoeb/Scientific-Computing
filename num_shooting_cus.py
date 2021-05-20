import numpy as np
import matplotlib.pyplot as plt
import phaseportraits as pp


def pred_prey(t, z, b):
    x, y = z
    a = 1.
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    return np.array((dxdt, dydt))

pred_ode = lambda t, u: pred_prey(t, u, 0.25)

pp.time_simulation(pred_ode, (0.32, 0.32), 100, "custom")

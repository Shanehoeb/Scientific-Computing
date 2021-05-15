import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from solve_ode import solve_ode

def pred_prey(x,y,b):
    a = 1
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    print(np.array((dxdt, dydt)))
    return np.array((dxdt, dydt))

sol = solve_ivp(fun)

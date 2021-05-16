import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from solve_ode import solve_ode
from math import nan

def pred_prey(t,z,b):
    x, y = z
    a = 1.
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    return np.array((dxdt, dydt))


sol = solve_ivp(lambda t,z:pred_prey(t,z,0.4), [0, 100], [4,1],dense_output=True)
plt.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t,z:pred_prey(t,z,0.4), [0, 100], [10,1],dense_output=True)
plt.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t,z:pred_prey(t,z,0.4), [0, 100], [5,5],dense_output=True)
plt.plot(sol.y[0, :], sol.y[1, :])
sol = solve_ivp(lambda t,z:pred_prey(t,z,0.4), [0, 100], [0.1,0],dense_output=True)
plt.plot(sol.y[0, :], sol.y[1, :])
plt.show()

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from solve_ode import solve_ode
from math import nan

def pred_prey(t,z):
    x, y = z
    a = 1.
    d = 0.1
    b = 0.2

    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    print(np.array((dxdt, dydt)))
    return np.array((dxdt, dydt))


sol = solve_ivp(pred_prey, [0, 100], [1,1],dense_output=True)
t = np.linspace(0, 100, 501)
z = sol.sol(t)
import matplotlib.pyplot as plt
plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()

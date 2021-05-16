import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
from solve_ode import solve_ode
from math import nan
import phaseportraits as pp

def pred_prey(t,z,b):
    x, y = z
    a = 1.
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    return np.array((dxdt, dydt))

pred_ode = lambda t, u: pred_prey(t, u, 0.26)

# Some orbits
(t, u) = pp.orbit(pred_ode, (0.8, 0.1), 100)
plt.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(pred_ode, (1., 0.2), 100)
plt.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(pred_ode, (0.5, 0.5), 100)
plt.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(pred_ode, (0.1, 0.5), 100)
plt.plot(u[0, :], u[1, :], "b-")
(t, u) = pp.orbit(pred_ode, (0.3, 1.0), 100)
plt.plot(u[0, :], u[1, :], "b-")


# V nullcline
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=0)
plt.plot(Vval, Nval, "g-")

# N nullcline - this fails
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=1)
#plt.plot(Vval, Nval, "r-")
# TODO : fix this nullcline problem

#For now hard code nullcline
x = np.linspace(-0.4,1,101)
y = np.linspace(-0.4,1,101)
plt.plot(x,y,"r-")


# Equilibria
u = pp.equilibrium(pred_ode, (-0.4,-0.4))
print(u)
plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (0.2, 0.2))
plt.plot(u[0], u[1], "k.")
print(u)
plt.show()
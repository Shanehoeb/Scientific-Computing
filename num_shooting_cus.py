import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt


def pred_prey(t, z, b):
    x, y = z
    a = 1.
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    return np.array((dxdt, dydt))


pred_ode = lambda t, u: pred_prey(t, u, 0.25)
pp.time_simulation(pred_ode, (0.32, 0.32), 100, "custom", plot=True)

# Single out periodic orbit
(t, u) = pp.orbit(pred_ode, (0.32, 0.32), 100, solver="custom", plot=True)


# Some orbits
(t, u) = pp.orbit(pred_ode, (0.8, 0.1), 100, solver="custom")
plt.plot(u[0], u[1], "r-")
(t, u) = pp.orbit(pred_ode, (1., 0.2), 100, solver="custom")
plt.plot(u[0], u[1], "g-")
(t, u) = pp.orbit(pred_ode, (0.5, 0.5), 100, solver="custom")
plt.plot(u[0], u[1], "k-")
(t, u) = pp.orbit(pred_ode, (0.1, 0.5), 100, solver="custom")
plt.plot(u[0], u[1], "b-")
(t, u) = pp.orbit(pred_ode, (0.3, 1.0), 100, solver="custom")
plt.plot(u[0], u[1], "y-")


# V null-cline
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=0)
plt.plot(Vval, Nval, "r-")


# N null-cline - this fails
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=1)
plt.plot(Vval, Nval, "g-")


# Equilibrium
u = pp.equilibrium(pred_ode, (-0.4, -0.4))
print(u)
plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (1., 0.))
print(u)
plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (-0.2, -0.1))
print(u)
plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (0.2, 0.2))
plt.plot(u[0], u[1], "k.")
print(u)
plt.title("Multiple Orbits, Null-clines & Equilibria")
plt.show()
import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def pred_prey(t, z, b):
    x, y = z
    a = 1.
    d = 0.1
    dxdt = x*(1-x) - (a*x*y)/(d+x)
    dydt = b*y*(1 - (y/x))
    return np.array((dxdt, dydt))


pred_ode = lambda t, u: pred_prey(t, u, 0.25)
a = pp.time_simulation(pred_ode, (0.32, 0.32), (0, 18.5), "custom", plot=True)
(t1, u1) = pp.orbit(pred_ode, (0.32, 0.32), (0, 18.5), solver="custom", plot=True)


pp.time_simulation(lambda t, u: pred_prey(t, u, 0.27), (0.32, 0.32), (0,100), "custom", plot=True)

# Single out periodic orbit
(t, u) = pp.orbit(pred_ode, (0.32, 0.32), (0,100), solver="custom", plot=True)


# Some orbits
(t, u) = pp.orbit(pred_ode, (0.8, 0.1), (0,100), solver="custom")
plt.plot(u[0], u[1], "r-")
(t, u) = pp.orbit(pred_ode, (1., 0.2), (0,100), solver="custom")
plt.plot(u[0], u[1], "g-")
(t, u) = pp.orbit(pred_ode, (0.5, 0.5), (0,100), solver="custom")
plt.plot(u[0], u[1], "k-")
(t, u) = pp.orbit(pred_ode, (0.1, 0.5), (0,100), solver="custom")
plt.plot(u[0], u[1], "m-")
(t, u) = pp.orbit(pred_ode, (0.3, 1.0), (0,100), solver="custom")
plt.plot(u[0], u[1], "y-")

# V null-cline
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=0)
plt.plot(Vval, Nval, "b-", linewidth=4)


# N null-cline - this fails
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=1)
plt.plot(Vval, Nval, "b-", linewidth=4)


# Equilibrium
u = pp.equilibrium(pred_ode, (-0.4, -0.4))

plt.plot(u[0], u[1], "r.", markersize=20)
u = pp.equilibrium(pred_ode, (1., 0.))

plt.plot(u[0], u[1], "r.", markersize=20)
u = pp.equilibrium(pred_ode, (-0.2, -0.1))

plt.plot(u[0], u[1], "r.", markersize=20)
u = pp.equilibrium(pred_ode, (0.2, 0.2))
plt.plot(u[0], u[1], "r.", markersize=20)

plt.title("Multiple Orbits, Nullclines & Equilibria")
red_patch = mpatches.Patch(color='red', label='Equilibria')
blue_patch = mpatches.Patch(color='blue', label='Nullclines')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

def bvp(t, z):
    x, y = z
    dxdt = y
    dydt = (-0.5*x) + (5/2)*y
    return np.array((dxdt, dydt))
u = np.array((6.,-1.))
t_span = (0., 3.)
guess = np.array((0., 3.))
f = lambda t, u: bvp(t, u)
pp.shoot_bvp(f, u, guess, t_span)

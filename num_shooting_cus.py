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
a = pp.time_simulation(pred_ode, (0.32, 0.32), (0, 1000), "custom", plot=True)
pred_ode = lambda t, u: pred_prey(t, u, 0.27)
b = pp.time_simulation(pred_ode, (0.32, 0.32), (0, 1000), "custom", plot=True)

pred_ode = lambda t, u: pred_prey(t, u, 0.25)
# Plot orbit examples
(t1, u1) = pp.orbit(pred_ode, (0.32, 0.32), (0, 100), solver="scipy", plot=False)
(t2, u2) = pp.orbit(pred_ode, (0.32, 0.32), (0, 100), solver="custom", plot=False)

fig,axs = plt.subplots(2, 2)
#Plot time simulations
axs[0, 0].plot(a[0], np.transpose(a[1]))
axs[0, 0].set_title("Scipy Solver Simulation")
axs[1, 0].plot(b[0], np.transpose(b[1]))
axs[1, 0].set_title("Custom Solver Simulation")

# Plot orbit examples
axs[0, 1].plot(u1[0], u1[1], "b-")
axs[0, 1].set_title("Scipy Orbit  IC : (0.32, 0.32)")
axs[1, 1].plot(u2[0], u2[1], "b-")
axs[1, 1].set_title("Custom Orbit IC : (0.32, 0.32)")
fig.tight_layout()
fig.show()

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
plt.plot(u[0], u[1], "b-")
(t, u) = pp.orbit(pred_ode, (0.3, 1.0), (0,100), solver="custom")
plt.plot(u[0], u[1], "y-")


# V null-cline
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=0)
plt.plot(Vval, Nval, "r-")


# N null-cline - this fails
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=1)
plt.plot(Vval, Nval, "g-")


# Equilibrium
u = pp.equilibrium(pred_ode, (-0.4, -0.4))

plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (1., 0.))

plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (-0.2, -0.1))

plt.plot(u[0], u[1], "k.")
u = pp.equilibrium(pred_ode, (0.2, 0.2))
plt.plot(u[0], u[1], "k.")

plt.title("Multiple Orbits, Null-clines & Equilibria")
plt.show()


def shoot_func(): # Function for predator prey ODE num shooting

    pass

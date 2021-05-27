import numpy as np
import ODEs as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Example function : Predator prey model

# Model Parameters
def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.26,
        "d": 0.1
    }
# Model function
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))

# Transoform so only dependant on time and u values.
pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())

print("Running Time Simulation...")
# Initial conditions example
initialu = (0.27, 0.27)
# Time period example
t_span = (0, 100)

# Time simulation with custom solver
t_array, sol = pp.time_simulation(pred_ode, initialu, t_span, solver="custom", plot=False)

# Plot Results in panel
fig, axs = plt.subplots(2, 2)
fig.set_size_inches(16, 10)
axs[0, 0].plot(t_array, sol[0], label="x")
axs[0, 0].plot(t_array, sol[1], label="y")
axs[0, 0].legend()
axs[0, 0].set_title("Custom Time Series Simulation for initial conditions %s" % ((0.27, 0.27) ,))
axs[0, 0].set(xlabel="t", ylabel='f(t,u)')

print("Computing orbits...")
# Orbit of the ODE
(t1, u1) = pp.orbit(pred_ode, initialu, t_span, solver="custom", plot=False)
# Plot orbit
axs[0, 1].plot(u1[0], u1[1], "b-")
axs[0, 1].set_title("Custom Orbit for initial conditions %s" % (initialu,))
axs[0, 1].set(xlabel="x", ylabel='y')

# Time simulation with scipy solver
t_array1, sol1 = pp.time_simulation(pred_ode, initialu, t_span, solver="scipy", plot=False)
axs[1, 0].plot(t_array1, sol1[0], label="x")
axs[1, 0].plot(t_array1, sol1[1], label="y")
axs[1, 0].legend()
axs[1, 0].set_title("Scipy Time Series Simulation for initial conditions %s" % ((0.27, 0.27) ,))
axs[1, 0].set(xlabel="t", ylabel='f(t,u)')

# Orbit of the ODE
(t2, u2) = pp.orbit(pred_ode, initialu, t_span, solver="scipy", plot=False)
# Plot orbit
axs[1, 1].plot(u2[0], u2[1], "b-")
axs[1, 1].set_title("Scipy Orbit for initial conditions %s" % (initialu,))
axs[1, 1].set(xlabel="x", ylabel='y')


fig.tight_layout()
fig.show()

# ODE analysis : Orbits, Nullclines & Equilibria
print("Computing more orbits...")
# Plot some orbits
(t, u) = pp.orbit(pred_ode, (0.8, 0.1), (0,100), solver="custom")
plt.plot(u[0], u[1], "c-")
(t, u) = pp.orbit(pred_ode, (1., 0.2), (0,100), solver="custom")
plt.plot(u[0], u[1], "g-")
(t, u) = pp.orbit(pred_ode, (0.5, 0.5), (0,100), solver="custom")
plt.plot(u[0], u[1], "k-")
(t, u) = pp.orbit(pred_ode, (0.1, 0.5), (0,100), solver="custom")
plt.plot(u[0], u[1], "m-")
(t, u) = pp.orbit(pred_ode, (0.3, 1.0), (0,100), solver="custom")
plt.plot(u[0], u[1], "y-")

print("Computing nullclines...")
# V null-cline
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=0)
plt.plot(Vval, Nval, "b-", linewidth=2)


# N null-cline - this fails
(Vval, Nval) = pp.nullcline(pred_ode, (-0.4, 1.1), index=1)
plt.plot(Vval, Nval, "r-", linewidth=2)

print("Computing equilibria...")
# Find system equilibria
equilibria = pp.find_equilibria(pred_ode, 2)
x_equil = []
y_equil = []
for element in equilibria:
    x_equil.append(element[0])
    y_equil.append(element[1])
plt.plot(x_equil, y_equil, "ko", label='Equilibria')

# Generate legends
red_patch = mpatches.Patch(color='red', label='Y Nullcline')
blue_patch = mpatches.Patch(color='blue', label='X Nullcline')
black_patch = mpatches.Patch(color='black', label='Equilibria')

plt.legend(handles=[blue_patch, red_patch, black_patch])
plt.title("Orbits, Nullclines & Equilibria of ODE")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print("Done !")
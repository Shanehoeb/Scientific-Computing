import matplotlib.pyplot as plt

from numerical_continuation import*
import numerical_shooting as ns
import ODEs as pp
import matplotlib.patches as mpatches

# Starting default values
def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.11,
        "d": 0.1
    }

# Predator Prey model as system of 1st order equations
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))

# Initial conditions example
initialu = (0.79, 0.18)
# Time period example
t_span = (0, 100)

# Set param values
pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())

# Time simulation with custom solver
t_array, sol = pp.time_simulation(pred_ode, initialu, t_span, solver="custom", plot=False)
fig, axs = plt.subplots(2)
axs[0].plot(t_array, sol[0], label="x")
axs[0].plot(t_array, sol[1], label="y")
axs[0].set(xlabel="t", ylabel='f(t, u)')
axs[0].legend(loc='lower right')
axs[0].set_title("Time Simulation of ODE")

# Numerical shooting to find initial conditions and period of limit cycle
solut = ns.shoot((0.79, 0.18, 31.), pred_ode, solver="custom", method="rk4", stepsize=0.125, deltat_max=20, index=0, plot=False)
period = solut[-1] # Period is last entry of solution array
initial_c = solut[:-1] # Initial values are rest of array

# Plot time simulation with found initial values
t_array1, sol1 = pp.time_simulation(pred_ode, initial_c, t_span, solver="custom", plot=False)

axs[1].plot(t_array1, sol1[0], label="x")
axs[1].plot(t_array1, sol1[1], label="y")
axs[1].plot([], [], "ro", label="Period T")
axs[1].set(xlabel="t", ylabel='f(t, u)')

t_1 = t_span[0]
t_2 = t_span[1]
points_T = []
while t_2 - t_1 > 0:
    # Draw points where function should be equal to initial value
    # with found period
    points_T.append(t_1)
    t_1 = t_1 + period
for point in points_T:
    axs[1].plot(point, initial_c[0], "ro")

axs[1].legend(loc='lower right')
axs[1].set_title("Time Simulation of ODE with Period using Numerical Shooting")

fig.tight_layout()
fig.show()

# Natural Parameter Continuation for parameter b in [0.1, 0.5]
x, u_j = natural_continuation((0.79, 0.18, 30.), pred_prey, (0.1, 0.5), 0.005, "b", defaults_pred_prey(),
                              solver="custom", method="rk4", stepsize=0.125, deltat_max=20, index=0, plot=False)
plt.plot(x, u_j, "ro", label="X value in limit cycle")
plt.xlabel("b")
plt.ylabel("x")
plt.title("Natural Parameter Continuation with Numerical Shooting")
plt.legend(loc="upper right")
plt.show()

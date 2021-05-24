import numpy as np
import phaseportraits as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.25,
        "d": 0.1
    }
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))



pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
a = pp.time_simulation(pred_ode, (0.32, 0.32), (0, 18.5), "custom", plot=True)
(t1, u1) = pp.orbit(pred_ode, (0.32, 0.32), (0, 18.5), solver="custom", plot=True)


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


import sympy as sm

r,c = sm.symbols('r, c')
R,C = pred_ode(np.nan, (r,c))
REqual = sm.Eq(R, 0)
CEqual = sm.Eq(C, 0)

# compute fixed points
equilibria = sm.solve( (REqual, CEqual), r, c)
print(equilibria)

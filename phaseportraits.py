from scipy.integrate import solve_ivp
from scipy.optimize import root
from math import nan
import matplotlib.pyplot as plt
from numerical_methods import*
import numpy as np


def scipy_solver(ode, initialu, duration, plot):
    sol = solve_ivp(ode, (0, duration), initialu)
    if plot:
        for i in range(len(sol.y)):
            plt.plot(sol.t, sol.y[i, :])
        plt.title("Time Series Simulation for initial conditions %d" % initialu)
        plt.show()
    return sol.t, sol.y


def my_solver(ode, initialu, duration, stepsize=0.005, method="rk4", deltat_max=2, plot=True):
    t_array, sol = solve_ode(ode, (0, duration), initialu, stepsize, method, deltat_max)
    sol = np.array(sol)
    solution = []
    for i in range(len(sol[0])):
        solution.append(sol[:, i])

        if plot:
            plt.plot(t_array, sol[:, i])
    if plot:
        plt.title("Time Series Simulation for initial conditions %s" % (initialu,))
        plt.show()

    return t_array, solution


def time_simulation(ode, initialu, duration, solver, method="rk4", stepsize=0.005, deltat_max=2, plot=True):
    if solver == "scipy":
        return scipy_solver(ode, initialu, duration, plot)
    elif solver == "custom":
        return my_solver(ode, initialu, duration, stepsize, method, deltat_max, plot)


def orbit(ode, initialu, duration,solver, method="rk4", stepsize=0.005, deltat_max=2, plot=False):
    if solver == "scipy":
        _, sol = scipy_solver(ode, initialu, duration, solver)
        if plot:
            plt.plot(sol[0], sol[1], "b-")
            plt.title("Periodic Orbit")
            plt.show()
        return _, sol
    elif solver == "custom":
        _, sol = my_solver(ode, initialu, duration, stepsize, method, deltat_max, plot=False)
        if plot:
            plt.plot(sol[0], sol[1], "b-")
            plt.title("Periodic Orbit  for initial conditions %s" % (initialu,))
            plt.show()
        return _, sol


def nullcline(ode, u0range, index=0, points=101):
    Vval = np.linspace(min(u0range), max(u0range), points)
    Nval = np.zeros((len(Vval), len(u0range)))
    for (i, V) in enumerate(Vval):
        result = root(lambda N: ode(nan, (V, N))[index], np.array([min(u0range), max(u0range)]))
        if result.success:
            Nval[i] = result.x
        else:
            Nval[i] = nan
    else:
        return (Vval, Nval)


def equilibrium(ode, initialu):
    result = root(lambda u: ode(nan, u), initialu)
    if result.success:
        return result.x
    else:
        return nan
    # TODO: Should I throw an error here instead?


def num_shoot(ode, func, initialu, t):
    step = 0
    # TODO : Numerical Shooting & Extension to all ODEs
    der = (func(t+step, initialu) - func(t, initialu))/ (t + step)
    result = root((der, func(t, initialu)), t)
    if result.success:
        return result.x
    else:
        return nan
    pass
# Num Shooting :
# Solve u_0 - f(u_0,T) = 0
# Define G(u_0) = [u_0 - f(u_o, T) , dx/dt(0)] = 0
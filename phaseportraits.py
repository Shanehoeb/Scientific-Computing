import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from math import nan
import matplotlib.pyplot as plt
from solve_ode import*


def scipy_solver(ode, initialu, duration, plot):
    sol = solve_ivp(ode, (0, duration), initialu)
    if plot:
        for i in range(len(sol.y)):
            plt.plot(sol.t, sol.y[i, :])
        plt.title("Time Series Simulation")
        plt.show()
    return sol.t, sol.y


def my_solver(ode, initialu, duration, stepsize, method, deltat_max, plot):
    t_array, sol = solve_ode(ode, (0, duration), initialu, stepsize, method, deltat_max)
    sol = np.array(sol)
    solution = []
    for i in range(len(sol[0])):
        solution.append(sol[:, i])

        if plot:
            plt.plot(t_array, sol[:, i])
    if plot:
        plt.title("Time Series Simulation")
        plt.show()

    return t_array, solution


def time_simulation(ode, initialu, duration, solver, method="rk4", stepsize=0.005, deltat_max=2, plot=True):
    if solver == "scipy":
        return scipy_solver(ode, initialu, duration, plot)
    elif solver == "custom":
        return my_solver(ode, initialu, duration,stepsize, method, deltat_max, plot)


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
            plt.title("Periodic Orbit")
            plt.show()
        return _, sol


def nullcline(ode, u0range, index=0, points=101):
    # TODO : Extend to more than 2D get rid of pval
    Vval = np.linspace(min(u0range), max(u0range), points)
    Nval = np.zeros(np.size(Vval))
    Pval = np.zeros(np.size(Vval))
    t = 0
    for (i, V) in enumerate(Vval):
        result = root(lambda N: ode(nan, (V, N))[index], [0, 5])
        if result.success:
            if len(result.x) > 1:
                Nval[i] = result.x[0]
                Pval[i] = result.x[1]
                t = 1
            else:
                Nval[i] = result.x[0]
        else:
            Nval[i] = nan
    if t == 1:
        return (Vval, np.column_stack((Nval,Pval)))
    else:
        return (Vval, Nval)


def equilibrium(ode, initialu):
    result = root(lambda u: ode(nan, u), initialu)
    if result.success:
        return result.x
    else:
        return nan
    # TODO: Should I throw an error here instead?

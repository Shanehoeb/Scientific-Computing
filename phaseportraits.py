from scipy.integrate import solve_ivp
from scipy.optimize import root, fsolve
from math import nan
import matplotlib.pyplot as plt
from numerical_methods import*
import numpy as np


def scipy_solver(ode, initialu, t_span, plot=False):
    sol = solve_ivp(ode, t_span, initialu)
    if plot:
        for i in range(len(sol.y)):
            plt.plot(sol.t, sol.y[i, :])
        plt.title("Time Series Simulation for initial conditions %s" % str(initialu))
        plt.show()
    return sol.t, sol.y


def my_solver(ode, initialu, t_span, stepsize=0.005, method="rk4", deltat_max=2, plot=True):
    t_array, sol = solve_ode(ode, t_span, initialu, stepsize, method, deltat_max)
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


def time_simulation(ode, initialu, duration, solver="custom", method="rk4", stepsize=0.005, deltat_max=2, plot=False):
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


def num_shot_bvp(equations, initial_guess, t_span, solver, index):
    # TODO : This function should return the numerical result for the boundary value we are looking for from given guess
    # Will serve as the low-level in our solving
    if solver == "scipy":
        sol = scipy_solver(equations, initial_guess, t_span, solver)
        val = sol[index][-1]
        return val
    if solver == "custom":
        pass

def secant(x, y, error):
    z1, z2 = x
    w1, w2 = y
    z1 = z2
    z2 = z2 + (z2 - z1) / (w2 - w1) * error
    return (z1, z2), w2


def shoot_bvp(f, u, guess, t_span, tol=0.01, solver="custom", points=101):
    #Solve boundary value problems with numerical shooting
    # INPUTS :
    # - f : function dy/dt = f(t,y). The function is transformed to first order system,
    #       returns array of size number of variables in the system.
    # - u : Array of Solutions at the right and left of the boundary
    # - guess : initial guess provided by user
    # - solver : custom or scipy
    # - t_span : tuple containing limits of time period
    # - points : nb of points to generate
    # - other parameters for custom
    # OUTPUT :
    # sol : array of solutions of bvp for t_span
    max_iteration = 100
    t_array = np.linspace(t_span[0], t_span[1], points)

    sol1 = time_simulation(f, np.array((u[0], guess[0])), t_span, solver=solver)
    val1 = sol1[1][-1][0]
    error1 = val1 - u[1]
    print(error1)
    for i in range(max_iteration):
        sol2 = time_simulation(f, np.array((u[0], guess[1])), t_span, solver=solver)
        val2 = sol2[1][-1][0]
        error2 = val2 - u[1]
        print(error2)
        if np.absolute(error2) < tol:
            print("Success!")
            break
        guess, val2 = secant(u, guess, error2)
        print(guess)




    print("Didnt converge")
    pass



# From Orbit Function !!
    # TODO : this function will iterate until it finds initial condition satisfying our boundary conditions
    # Change guess with Newton iteration for roots

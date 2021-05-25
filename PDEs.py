import scipy.sparse
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.sparse import diags
import numpy as np


def default_heat_params():
    return{
        "L": 1.,
        "kappa": 1.,
        "T": 0.5
    }

def u_I(x,params):
    # initial temperature distribution
    y = np.sin(pi*x/params["L"])
    return y

def u_exact(x,t,params):
    # the exact solution
    y = np.exp(-params["kappa"]*(pi**2/params["L"]**2)*t)*np.sin(pi*x/params["L"])
    return y


def calc_lambda(deltat,deltax,params):
    return params["kappa"] * deltat / (deltax ** 2)


def setup_env(params,mx,mt):
    # Set up the numerical environment variables
    x = np.linspace(0, params["L"], mx + 1)  # mesh points in space
    t = np.linspace(0, params["T"], mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step
    return x,t,deltax,deltat,u_j,u_jp1


def component_feuler(u_I, params, mx=10, mt=100, plot=False):
    # Set up the numerical environment variables
    x, t, deltax, deltat, u_j, u_jp1 = setup_env(params, mx, mt)
    lmbda = calc_lambda(deltat,deltax,params) # mesh fourier number

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i],params)

    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        for i in range(1, mx):
            u_jp1[i] = u_j[i] + lmbda * (u_j[i - 1] - 2 * u_j[i] + u_j[i + 1])
            # Boundary conditions
        u_jp1[0] = 0
        u_jp1[mx] = 0

        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    if plot:
        plt.plot(x, u_j, 'ro', label='num')
        plt.show()

    return x, u_j

def vectorised_feuler(u_I, params, mx=60, mt=8000, plot=False):
    x, t, deltax, deltat, u_j, _ = setup_env(params, mx, mt)
    lmbda = calc_lambda(deltat, deltax, params) # mesh fourier number
    results = []
    # Set initial condition
    for i in range(1, mx):
        u_j[i] = u_I(x[i], params)

    if lmbda > 0.5:
        print("Change mx and mt to proceed !")
        return
    n = round(mx+1)
    k = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()
    for i in range(mt):
        u_j = A.dot(u_j)
        results.append(u_j)

    if plot:
        for i in range(len(results)):
            plt.plot(x, results[i], 'ro', label='num')
        plt.show()
    return x, results


def vectorised_back_euler(u_I, params, mx=60, mt=8000, plot=False):
    x, t, deltax, deltat, u_j, _ = setup_env(params, mx, mt)
    lmbda = calc_lambda(deltat, deltax, params) # mesh fourier number
    results = []
    # Set initial condition
    for i in range(1, mx):
        u_j[i] = u_I(x[i], params)

    n = round(mx + 1)
    k = np.array([-lmbda * np.ones(n - 1), 2*lmbda +np.ones(n), -lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()


    pass

def crank_nicholson():
    pass

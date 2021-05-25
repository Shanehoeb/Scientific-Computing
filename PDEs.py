import scipy.sparse
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np

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


def component_feuler(u_I, params, mx=60, mt=8000, plot=False):
    # Set up the numerical environment variables
    x, t, deltax, deltat, u_j, u_jp1 = setup_env(params, mx, mt)
    lmbda = calc_lambda(deltat,deltax,params) # mesh fourier number

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i], params)

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


def forward_euler(lmbda, u_j, mx, mt):
    n = round(mx - 1)
    k = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()
    for i in range(mt):
        past_u_j = u_j[1:mx]
        sol = A.dot(past_u_j)
        u_j = [0]
        for element in sol:
            u_j.append(element)
        u_j.append(0)
    return u_j


def backwards_euler(lmbda, u_j, mx, mt):
    n = round(mx - 1)
    k = np.array([-lmbda * np.ones(n - 1), 2 * lmbda + np.ones(n), -lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset, format='csr')
    for i in range(mt):
        past_u_j = u_j[1:mx]
        sol = spsolve(A, past_u_j)
        u_j = [0]
        for element in sol:
            u_j.append(element)
        u_j.append(0)
    return u_j


def crank_nicholson(lmbda, u_j, mx, mt):
    n = round(mx - 1)

    k = np.array([(-lmbda / 2) * np.ones(n - 1), lmbda + np.ones(n), (-lmbda / 2) * np.ones(n - 1)],
                 dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset, format='csr')

    k2 = np.array([(lmbda / 2) * np.ones(n - 1), np.ones(n) - lmbda, (lmbda / 2) * np.ones(n - 1)],
                  dtype=np.dtype(object))
    B = diags(k2, offset, format='csr')
    for i in range(mt):
        past_u_j = u_j[1:mx]
        eq = B.dot(past_u_j)
        sol = spsolve(A, eq)
        u_j = [0]
        for element in sol:
            u_j.append(element)
        u_j.append(0)
    return u_j


def pde_solver(u_I, params, mx=100, mt=100, method="ck", plot=False):
    x, t, deltax, deltat, u_j, _ = setup_env(params, mx, mt)
    lmbda = calc_lambda(deltat, deltax, params)  # mesh fourier number
    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i], params)
    u_j[0] = 0
    u_j[mx] = 0

    if method == "ck":
        u_j = crank_nicholson(lmbda, u_j, mx, mt)
    elif method == "f-euler":
        if lmbda < 0.5:
            u_j = forward_euler(lmbda, u_j, mx, mt)
        else:
            print("Leads to unstable solutions, change grid properties.")
            return
    elif method == "b-euler":
        u_j = crank_nicholson(lmbda, u_j, mx, mt)

    if plot:
        plt.plot(x, u_j, 'ro', label='num')
        plt.show()

    return x, u_j

import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np


def calc_lambda(deltat, deltax, params):
    """Calculates the value of lambda for the diffusion PDE.

       Parameters
       ---------
       deltat : float
                Time step size.

       deltax : float
                Space step size.

       params : dict
                Parameters of the PDE.

       Returns
       ---------
       Value of lambda (float) for grid of finite difference method."""
    return params["kappa"] * deltat / (deltax ** 2)


def setup_env(mx, mt, params):
    """Initialises variables necessary for solving the PDE

       Parameters
       ---------
       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       params : dict
                Parameters of the PDE.

       Returns
       ---------
       x : array, size(mx+1)
           Array of equally-spaced points in space.

       t : array, size(mx+1)
           Array of equally-spaced points in time.

       deltax : float
                Step size in space.

       deltat : float
                Step size in time

       u_j, u_jp1 : arrays
                    Initialise arrays to store solutions.
       """
    # Set up the numerical environment variables
    x = np.linspace(0, params["L"], mx + 1)  # mesh points in space
    t = np.linspace(0, params["T"], mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step
    return x, t, deltax, deltat, u_j, u_jp1


def zero_case(past_u_j, A):
    """Forward Euler step for 0 valued boundary conditions.

       Parameters
       ---------
       past_u_j : array, size(mx-1,)
                  Initial vector solution of PDE for time t.

       A : array, size(mx-1, mx-1)
           Forward Euler tridiagonal matrix.

       Returns
       ---------
       u_j : array, size(mx+1,)
             Solution vector at time t+1.

    """
    # Compute solution
    sol = A.dot(past_u_j)
    # Add boundary values
    u_j = [0]
    for element in sol:
        u_j.append(element)
    u_j.append(0)
    return u_j


def dirichlet_case(past_u_j, A, lmbda, mt, bound_funcs):
    """Forward Euler step for Dirichlet boundary conditions.

       Parameters
       ---------
       past_u_j : array, size(mx-1,)
                  Initial vector solution of PDE for time t.

       A : array, size(mx-1, mx-1)
           Forward Euler tridiagonal matrix.

       lmbda : float
               Lambda value of diffusive heat equation for chosen resolution.

       mt : int
            Time index.

       bound_funcs : tuple, size(2,)
                    Tuple of functions describing the conditions at the boundaries
                    for Dirichlet case.

       Returns
       ---------
       u_j : array, size(mx+1,)
             Solution vector at time t+1.

    """
    # Create vector
    dir_vec = np.zeros(len(past_u_j))
    # Extract boundary functions
    p, q = bound_funcs
    # Compute boundary values
    dir_vec[0] = p(mt)
    dir_vec[-1] = q(mt)
    # Compute solution
    sol = A.dot(past_u_j) + lmbda * np.array(dir_vec)
    # Add boundary values
    u_j = [p(mt)]
    for element in sol:
        u_j.append(element)
    u_j.append(q(mt))
    return u_j


def neumann_case(past_u_j, A, lmbda, mt, deltax, bound_funcs):
    """Forward Euler step for Neumann boundary conditions.

       Parameters
       ---------
       past_u_j : array, size(mx+1,)
                  Initial vector solution of PDE for time t.

       A : array, size(mx+1, mx+1)
           Forward Euler tridiagonal matrix.

       lmbda : float
               Lambda value of diffusive heat equation for chosen resolution.

       mt : int
            Time index.

       deltax : float
                Chosen step size in space.

       bound_funcs : tuple, size(2,)
                    Tuple of functions describing the conditions at the boundaries
                    for Neumann case (rate of diffusion).

       Returns
       ---------
       u_j : array, size(mx+1,)
             Solution vector at time t+1.

    """

    dir_vec = np.zeros(len(past_u_j))
    # Extract boundary functions
    p, q = bound_funcs
    # Add boundary info
    dir_vec[0] = -p(mt)
    dir_vec[-1] = q(mt)
    # Solve
    sol = A.dot(past_u_j) + 2*lmbda * deltax*np.array(dir_vec)
    u_j = []
    for element in sol:
        u_j.append(element)
    # Return solution vector
    return u_j

def periodic_case(past_u_j, A):
    """Forward Euler step for periodic boundary conditions.

       Parameters
       ---------
       past_u_j : array, size(mx,)
                  Initial vector solution of PDE for time t.

       A : array, size(mx, mx)
           Forward Euler tridiagonal matrix.

       Returns
       ---------
       u_j : array, size(mx+1,)
             Solution vector at time t+1.

    """
    # Solve
    sol = A.dot(past_u_j)
    u_j = []
    for element in sol:
        u_j.append(element)
    # Return solution vector
    u_j = np.append(u_j, u_j[-1])
    return u_j



def forward_euler(lmbda, u_j, mx, mt, deltax, deltat, bound_funcs=None, boundary_conds="zero", pde_type="", heat_func=None):
    """Approximates the PDE solution using forward euler finite difference.

       Parameters
       ---------
       lmbda : float
               Value of lambda for grid of finite difference method.

       u_j : array, size(mx+1,)
             Initial vector solution of PDE for t = 0.
       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       deltax : float
                Step size in space.

       deltat : float
                Step size in time

       bound_funcs : tuple, size(2,)
                    Tuple of functions used for non-homogeneous Dirichlet or Neumann
                    boundary conditions. Functions correspond to values/derivatives
                    at boundaries. Default is None.

       boundary_conds : string
                        String specifying the PDE problem's form. Available methods for
                        zero valued boundaries ("zero"), Dirichlet boundary conditions
                        ("dirichlet"), Neumann boundary conditions ("neumann") and
                        periodic boundary conditions ("periodic"). Default is "zero".

       pde_type : string
                  String specifying the type of PDE. Default is regular diffusive heat.
                  Available methods are addtional heat source F(x,t), used by passing
                  "heat_source" as pde_type.

       heat_func : callable
                   Function F(x,t) describing the additional heat source in the problem for
                   heat_source PDE type. Default is None
       Returns
       ---------
       Solution vector at the last point in time array using forward euler."""
    # Create appropriate tri-diagonal matrix
    n = round(mx - 1)
    k = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()
    # Calculate for each time step
    for i in range(mt):
        past_u_j = u_j[1:mx]
        # Zero boundary valued case
        if boundary_conds == "zero":
            u_j = zero_case(past_u_j, A)
        # Dirichlet BCs
        if boundary_conds == "dirichlet":
            u_j = dirichlet_case(past_u_j, A, lmbda, mt, bound_funcs)
        # Neumann BCs
        if boundary_conds == "neumann":
            n = round(mx + 1)
            k = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)],
                         dtype=np.dtype(object))
            offset = [-1, 0, 1]
            A = diags(k, offset).toarray()
            u_j = neumann_case(u_j, A, lmbda, mt, deltax, bound_funcs)
        # Periodic BCs
        if boundary_conds == "periodic":
            past_u_j = u_j[:mx]
            n = round(mx)
            k = np.array([lmbda * np.ones(n - 1), np.ones(n) - 2 * lmbda, lmbda * np.ones(n - 1)],
                         dtype=np.dtype(object))
            offset = [-1, 0, 1]
            A = diags(k, offset).toarray()
            A[-1][0] = lmbda
            A[0][-1] = lmbda
            u_j = periodic_case(past_u_j, A)
        # Heat source i
        if pde_type == "heat_source":
            def heat_source(f, x, t, deltat):
                return deltat * f(x, t)
            heat_vec = np.zeros(len(u_j))
            for j in range(len(heat_vec)):
                heat_vec[j] = heat_source(heat_func, j, mx*deltat, deltat)
            u_j = np.array(u_j) + heat_vec
    # Return final vector
    return u_j


def backwards_euler(lmbda, u_j, mx, mt):
    """Approximates the PDE solution using backward euler finite difference. Only for regular
        heat diffusion equation with 0 valued boundaries.

       Parameters
       ---------
       lmbda : float
               Value of lambda for grid of finite difference method.

       u_j : array, size(mx+1,)
             Initial vector solution of PDE for t = 0.
       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       Returns
       ---------
       Solution vector at the last point in time array."""
    # Create appropriate tri-diagonal matrix
    n = round(mx - 1)
    k = np.array([-lmbda * np.ones(n - 1), 2 * lmbda + np.ones(n), -lmbda * np.ones(n - 1)], dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset, format='csr')
    # Solve for each time step
    for i in range(mt):
        past_u_j = u_j[1:mx]
        sol = spsolve(A, past_u_j)
        u_j = [0]
        for element in sol:
            u_j.append(element)
        u_j.append(0)
    # Return final vector
    return u_j


def crank_nicholson(lmbda, u_j, mx, mt):
    """Approximates the PDE solution using Crank-Nicholson method for finite difference.
        Only for regular heat diffusion equation with 0 valued boundaries.

       Parameters
       ---------
       lmbda : float
               Value of lambda for grid of finite difference method.

       u_j : array, size(mx+1,)
             Initial vector solution of PDE for t = 0.

       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       Returns
       ---------
       Solution vector at the last point in time array."""
    # Create appropriate tri-diagonal matrices
    n = round(mx - 1)
    k = np.array([(-lmbda / 2) * np.ones(n - 1), lmbda + np.ones(n), (-lmbda / 2) * np.ones(n - 1)],
                 dtype=np.dtype(object))
    offset = [-1, 0, 1]
    A = diags(k, offset, format='csr')

    k2 = np.array([(lmbda / 2) * np.ones(n - 1), np.ones(n) - lmbda, (lmbda / 2) * np.ones(n - 1)],
                  dtype=np.dtype(object))
    B = diags(k2, offset, format='csr')
    # Solve for each time step
    for i in range(mt):
        past_u_j = u_j[1:mx]
        eq = B.dot(past_u_j)
        sol = spsolve(A, eq)
        u_j = [0]
        for element in sol:
            u_j.append(element)
        u_j.append(0)
    # Return final vector
    return u_j


def pde_solver(u_I, params, mx=100, mt=100, bound_funcs=None, boundary_conds="zero", pde_type="", method="f-euler", heat_func=None, plot=False):
    """Top level PDE solver. Returns PDE solution values approximated with chosen
        method.

       Parameters
       ---------
       u_I : func
             Function describing initial temperature distribution

       params : dict
                Parameters of the PDE.

       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       bound_funcs : tuple, size(2,)
                    Tuple of functions used for non-homogeneous Dirichlet or Neumann
                    boundary conditions. Functions correspond to values/derivatives
                    at boundaries. Default is None.

       boundary_conds : string
                        String specifying the PDE problem's form. Available methods for
                        zero valued boundaries ("zero"), Dirichlet boundary conditions
                        ("dirichlet"), Neumann boundary conditions ("neumann") and
                        periodic boundary conditions ("periodic"). Default is "zero".

       pde_type : string
                  String specifying the type of PDE. Default is regular diffusive heat.
                  Available methods are addtional heat source F(x,t), used by passing
                  "heat_source" as pde_type.

       method : str
                Method to solve the PDE (Forward Euler : "f-euler", Backward Euler :
                "b-euler" and Crank-Nicholson : "ck")

       heat_func : callable
                   Function F(x,t) describing the additional heat source in the problem for
                   heat_source PDE type. Default is None

       plot: bool
             Boolean value passed by the user; if True, a plot of the calculated
             solutions of the PDE will be generated.

       Returns
       ---------
       x : Values of space used for calculations.

       u_j : Solution vector at the last point in time array.
        """
    x, t, deltax, deltat, u_j, _ = setup_env(mx, mt, params)
    lmbda = calc_lambda(deltat, deltax, params)  # mesh fourier number
    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i], params)
    u_j[0] = 0
    u_j[mx] = 0
    # Crank-Nicholcon Case
    if method == "ck":
        u_j = crank_nicholson(lmbda, u_j, mx, mt)
    # Forward Euler Case
    elif method == "f-euler":
        if lmbda < 0.5:
            u_j = forward_euler(lmbda, u_j, mx, mt, deltax, deltat,
                                bound_funcs=bound_funcs, boundary_conds=boundary_conds, pde_type=pde_type, heat_func=heat_func)
        else:
            # Stability Issues
            print("Leads to unstable solutions, change grid properties.")
            return
    # Backward Euler Case
    elif method == "b-euler":
        u_j = crank_nicholson(lmbda, u_j, mx, mt)
    # Plot
    if plot:
        plt.plot(x, u_j, 'ro', label='num')
        plt.show()

    return x, u_j


# Function constructed from Martin's file
def component_feuler(u_I, params, mx=60, mt=8000, plot=False):
    """

       Parameters
       ---------
       u_I : func
             Function describing initial temperature distribution

       params : dict
                Parameters of the PDE.

       mx : int
            Number of grid points in space.

       mt : int
            Number of grid points in time.

       plot: bool
             Boolean value passed by the user; if True, a plot of the calculated
             solutions of the PDE will be generated.

       Returns
       ---------
       Solution vector at the last point in time array using forward euler step.
    """

    # Set up the numerical environment variables
    x, t, deltax, deltat, u_j, u_jp1 = setup_env(mx, mt, params)
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
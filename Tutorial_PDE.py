import matplotlib.pyplot as plt

from PDEs import*
from math import pi

# PDE parameters
def default_heat_params():
    return{
        "L": 1.,
        "kappa": 1.,
        "T": 0.5
    }

# Initial function
def u_I(x,params):
    # initial temperature distribution
    y = np.sin(pi*x/params["L"])
    return y

def u_exact(x,t, params):
    # the exact solution
    y = np.exp(-params['kappa']*(pi**2/params['L']**2)*t)*np.sin(pi*x/params['L'])
    return y

params = default_heat_params()

# Finite difference using forward euler for 0 valued boundary conditions
x, u_j = pde_solver(u_I, method="ck", params=default_heat_params(), mx=1000, mt=1000, plot=False)
plt.plot(x, u_j, 'ro', label='num')
xx = np.linspace(0,params['L'], 250)
plt.plot(xx,u_exact(xx, params['T'], params), 'b-', label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Crank-Nicholson")
plt.show()


# Finite difference using Crank-Nicholson for 0 valued boundary conditions
x, u_j = pde_solver(u_I, params=default_heat_params(), mx=60, mt=8000, plot=False)
plt.plot(x, u_j, 'ro', label='num')
xx = np.linspace(0,params['L'], 250)
plt.plot(xx,u_exact(xx, params['T'], params), 'b-', label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler")
plt.show()

# Backward Euler also available using method="b-euler"

# Non-Homogeneous Dirichlet Boundary Conditions


def p(t):
    return t/100


def q(t):
    return t/100


x, u_j = pde_solver(u_I, boundary_conds="dirichlet", bound_funcs=(p, q), params=default_heat_params(),
                    mx=60, mt=8000, plot=False)
plt.plot(x, u_j, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler Dirichlet")
plt.show()


# Neumann Boundary Conditions

x, u_j = pde_solver(u_I, boundary_conds="neumann", bound_funcs=(p, q), params=default_heat_params(),
                    mx=60, mt=8000, plot=False)
plt.plot(x, u_j, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler Neumann")
plt.show()


# Periodic boundary conditions

x, u_j = pde_solver(u_I, boundary_conds="periodic", params=default_heat_params(),
                    mx=60, mt=8000, plot=False)
plt.plot(x, u_j, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler Periodic")
plt.show()


# Add heat sources inside the domain

def heat_func(x, t):
    return 2*x + t

x, u_j = pde_solver(u_I, boundary_conds="zero", pde_type="heat_source",heat_func=heat_func, params=default_heat_params(),
                    mx=60, mt=8000, plot=False)

plt.plot(x, u_j, 'ro', label='num')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler with Heat Source")
plt.show()




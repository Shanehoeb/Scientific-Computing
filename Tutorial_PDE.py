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
print("Solving basic heat equation (forward Euler)...")
x, u_j = pde_solver(u_I, params=default_heat_params(), mx=30, mt=1000, plot=False)
fig, axs = plt.subplots(2)
axs[0].plot(x, u_j, 'ro', label='num')
xx = np.linspace(0,params['L'], 250)
axs[0].plot(xx,u_exact(xx, params['T'], params), 'b-', label='exact')
axs[0].set(xlabel="x", ylabel='u(x,0.5)')
axs[0].legend(loc='upper right')
axs[0].set_title("Heat Diffusion PDE Forward Euler")




# Finite difference using Backward Euler for 0 valued boundary conditions
print("Solving basic heat equation (backward Euler)...")
x, u_j = pde_solver(u_I, method="b-euler", params=default_heat_params(), mx=100, mt=100, plot=False)
axs[1].plot(x, u_j, 'ro', label='num')
xx = np.linspace(0,params['L'], 250)
axs[1].plot(xx,u_exact(xx, params['T'], params), 'b-', label='exact')
axs[1].set(xlabel="x", ylabel='u(x,0.5)')

axs[1].legend(loc='upper right')
axs[1].set_title("Heat Diffusion PDE Backward Euler")
fig.tight_layout()
fig.show()


# Finite difference using Crank-Nicholson for 0 valued boundary conditions
print("Solving basic heat equation (Crank-Nicholson)...")
x, u_j = pde_solver(u_I, method="ck", params=default_heat_params(), mx=100, mt=100, plot=False)
plt.plot(x, u_j, 'ro', label='num')
xx = np.linspace(0,params['L'], 250)
plt.plot(xx,u_exact(xx, params['T'], params), 'b-', label='exact')
plt.xlabel('x')
plt.ylabel('u(x,0.5)')
plt.legend(loc='upper right')
plt.title("Heat Diffusion PDE Forward Euler Crank-Nicholson")
plt.show()



# Non-Homogeneous Dirichlet Boundary Conditions
print("Solving Dirichlet BCs heat equation (forward Euler)...")

def p(t):
    return t/100


def q(t):
    return t/100


x, u_j = pde_solver(u_I, boundary_conds="dirichlet", bound_funcs=(p, q), params=default_heat_params(),
                    mx=60, mt=8000, plot=False)

fig, axs = plt.subplots(2, 2)
fig.set_size_inches(16, 10)
axs[0, 0].plot(x, u_j, 'ro', label='num')
axs[0, 0].set(xlabel="x", ylabel='u(x,0.5)')
axs[0, 0].legend(loc='upper right')
axs[0, 0].set_title("Heat Diffusion PDE Forward Euler Dirichlet")


# Neumann Boundary Conditions
print("Solving Neumann BCs heat equation (forward Euler)...")
x, u_j = pde_solver(u_I, boundary_conds="neumann", bound_funcs=(p, q), params=default_heat_params(),
                    mx=60, mt=8000, plot=False)
axs[0, 1].plot(x, u_j, 'ro', label='num')
axs[0, 1].set(xlabel="x", ylabel='u(x,0.5)')
axs[0, 1].legend(loc='upper right')
axs[0, 1].set_title("Heat Diffusion PDE Forward Euler Neumann")


# Periodic boundary conditions
print("Solving periodic BCs heat equation (forward Euler)...")
x, u_j = pde_solver(u_I, boundary_conds="periodic", params=default_heat_params(),
                    mx=60, mt=8000, plot=False)
axs[1, 0].plot(x, u_j, 'ro', label='num')
axs[1, 0].set(xlabel="x", ylabel='u(x,0.5)')
axs[1, 0].legend(loc='upper right')
axs[1, 0].set_title("Heat Diffusion PDE Forward Euler Periodic")


# Add heat sources inside the domain
print("Solving  heat equation with additional heat source (forward Euler)...")
def heat_func(x, t):
    return 2*x + t

x, u_j = pde_solver(u_I, boundary_conds="zero", pde_type="heat_source",heat_func=heat_func, params=default_heat_params(),
                    mx=60, mt=8000, plot=False)

axs[1, 1].plot(x, u_j, 'ro', label='num')
axs[1, 1].set(xlabel="x", ylabel='u(x,0.5)')
axs[1, 1].legend(loc='upper right')
axs[1, 1].set_title("Heat Diffusion PDE Forward Euler with Heat Source")
fig.tight_layout()
fig.show()

print("Done!")
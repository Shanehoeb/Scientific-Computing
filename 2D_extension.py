import numpy as np
import matplotlib.pyplot as plt
from solve_ode import solve_ode

t_0 = 0
x0 = 0
v0 = 1
X0 = [x0, v0]

# Solve from t=0 to t=10 and get 200 equally spaced points in the output
t_array = np.linspace(0, 10, 201)

deltat_max = 2

error_list_rk4 = []
error_list_euler = []
stepsize = np.longdouble(0.005)


def f_shm(t,X):
    x, v = X
    dxdt = v
    dvdt = -x
    dXdt = np.array([dxdt, dvdt])
    return dXdt


method = 'euler'
t_span = (0, 100)
# X_solution is a matrix with 200 rows and 2 columns. The first column is
# x and the other is v.

_, X_solution = solve_ode(f_shm, t_span, X0, stepsize,  method, deltat_max)

x = np.array(X_solution)[:, 0]
y = np.array(X_solution)[:, 1]
 # Create a figure with two plotting axes side by side:
fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_axes([0.58, 0.15, 0.35, 0.7])
ax2 = fig.add_axes([0.08, 0.15, 0.35, 0.7])

# Timeseries plot
ax1.set_title('Time series: $x, v$ against $t$')
ax1.plot(t_array, x, color='green', linewidth=2, label=r'$x$')
ax1.plot(t_array, y, color='blue', linewidth=2, label=r'$v$')
ax1.set_yticks([-1, 0, 1])
ax1.set_xlabel(r'$t$')
ax1.set_xticks([0, np.pi, 2*np.pi, 3*np.pi])
ax1.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$'])
ax1.grid()
ax1.legend()

# Phasespace plot
ax2.set_title('Phase space: $v$ against $x$')
ax2.plot(x, y, linewidth=2, color='red')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$v$', rotation=0)
ax2.set_xticks([-1, 0, 1])
ax2.set_yticks([-1, 0, 1])
ax2.grid()
plt.show()

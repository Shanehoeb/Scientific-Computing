import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import*
import time


#Params
stepsize_list = np.logspace(-5, -2.5, 201, dtype='float64')
    
x_0 = 1
deltat_max = 2

error_list_euler = []
error_list_rk = []


def func(t, x):
    return x

#Timing for similar errors
start = time.process_time()
sol = solve_ode(func, (0., 1.), x_0, 0.125, "euler", deltat_max)
print("Time for Euler method: %s"%str(time.process_time() - start))
start = time.process_time()
sol = solve_ode(func, (0., 1.), x_0, 0.125, "rk4", deltat_max)
print("Time for RK4 method: %s"%str(time.process_time() - start))

print("Running...")
for stepsize in stepsize_list:
    sol = solve_ode(func, (0., 1.), x_0, stepsize, "euler", deltat_max)
    error = np.absolute(np.exp(1)-sol[1][-1])
    error_list_euler.append(error)

for stepsize in stepsize_list:
    sol = solve_ode(func, (0., 1.), x_0, stepsize, "rk4", deltat_max)
    error = np.absolute(np.exp(1)-sol[1][-1])
    error_list_rk.append(error)


fig = plt.figure()
ax = plt.gca()
plt.scatter(stepsize_list, error_list_euler, 5, c="r")
plt.scatter(stepsize_list, error_list_rk, 5, c="b")
ax.set_yscale('log')
ax.set_xscale('log')
plt.title("Log-Log Plot of Error vs Stepsize")
plt.xlabel("Stepsize")
plt.ylabel("Error")
plt.legend(["Euler", "Runge-Kutta"])
plt.show()

fig = plt.figure()
ax = plt.gca()
plt.scatter(stepsize_list, error_list_euler, 5, c="r")
ax.set_yscale('log')
ax.set_xscale('log')
plt.title("Log-Log Plot of Error vs Stepsize")
plt.xlabel("Stepsize")
plt.ylabel("Error")
plt.legend(["Euler"])
plt.show()

fig = plt.figure()
ax = plt.gca()
plt.scatter(stepsize_list, error_list_rk, 5, c="b")
ax.set_yscale('log')
ax.set_xscale('log')
plt.title("Log-Log Plot of Error vs Stepsize")
plt.xlabel("Stepsize")
plt.ylabel("Error")
plt.legend(["Runge-Kutta"])
plt.show()

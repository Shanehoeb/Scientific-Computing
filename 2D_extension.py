from ODEs import*

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


method = 'rk4'
t_span = (0, 100)
# X_solution is a matrix with 200 rows and 2 columns. The first column is
# x and the other is v.

_, X_solution = my_solver(f_shm, X0, (0,100))
# Single out periodic orbit
(t, u) = orbit(f_shm, (0.32, 0.32), (0,100), solver="custom", plot=True)




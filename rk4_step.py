def rk4_step(stepsize,func,x_n,t_n):
     #Performs a single 4th order Runge-Kutta step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize,func the ODE
    # Output : Updated pair of values after Runge-Kutta step of stepsize
    k_1 = func(x_n,t_n)
    k_2 = func(x_n + stepsize*k_1*1/2, t_n + stepsize * 1/2)
    k_3 = func(x_n + stepsize*k_2*1/2, t_n + stepsize * 1/2)
    k_4 = func(x_n + stepsize*k_3, t_n + stepsize)
    return x_n + 1/6*stepsize*(k_1 + 2*k_2 + 2*k_3 + k_4),t_n + stepsize

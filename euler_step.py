def euler_step(stepsize,func,x_n,t_n):
    #Performs a single euler step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize,func the ODE
    # Output : Updated pair of values after euler step of stepsize
    return x_n + stepsize*func(x_n,t_n), t_n + stepsize

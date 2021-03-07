def euler_step(stepsize,x_n,t_n):
    #Performs a single euler step for a value of x, t and a stepsize
    # Input : x_n,t_n pair of coordinates, stepsize
    # Output : Updated pair of values after euler step of stepsize
    return x_n + stepsize*x_n, t_n + stepsize
    

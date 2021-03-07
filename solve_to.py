from euler_step import euler_step

def solve_to(x_1,t_1,t_2,stepsize,deltat_max,func):
    #solves from x_1,t_1 to x_2,t_2 in steps no bigger than deltat_max
    # Input : x_1,t_1 pair of coordinates, t_2 time at which we want to approximate,
    #         stepsize of the euler step and the maximum difference between 
    #         input and output time deltat_max
    if t_2 - t_1 < deltat_max:
        while t_1 < t_2:
            delta_t = t_2 - t_1
            if delta_t < stepsize:
                stepsize = delta_t
            x_1,t_1 = euler_step(stepsize,func, x_1, t_1)  
        return x_1, t_1
    else:
        print("Difference is too big ! Delta t max exceeded.")

from euler_step import euler_step
from rk4_step import rk4_step

def solve_to(X,t_1,t_2,stepsize,deltat_max,func,method):
    #solves from x_1,t_1 to x_2,t_2 in steps no bigger than deltat_max
    # Input : x_1,t_1 pair of coordinates, t_2 time at which we want to approximate,
    #         stepsize of the euler step and the maximum difference between 
    #         input and output time deltat_max
    if t_2 - t_1 < deltat_max:
        while t_1 < t_2:
            delta_t = t_2 - t_1
            if delta_t < stepsize:
                stepsize = delta_t
            if method == 'euler':
                X = euler_step(stepsize, func, X, t_1)[0]
            if method == 'rk4':
                X = rk4_step(stepsize, func, X, t_1)[0]

            t_1 += stepsize
        return X, t_1
    else:
        print("Difference is too big ! Delta t max exceeded.")

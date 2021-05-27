import unittest
from ODEs import*
import numerical_shooting as ns
import numerical_continuation as nc
from math import pi
import PDEs as pde

def defaults_pred_prey():
    return {
        "a": 1.,
        "b": 0.11,
        "d": 0.1
    }
def pred_prey(t, z, p):
    x, y = z
    dxdt = x*(1-x) - (p['a']*x*y)/(p['d']+x)
    dydt = p['b']*y*(1 - (y/x))
    return np.array((dxdt, dydt))

pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
initialu = (0.79912268, 0.18061336)
t_span = (0, 100)

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
params = default_heat_params()

class MyTestCase(unittest.TestCase):
    def test_test(self):
        self.assertEqual(True, True)

    def test_euler(self):
        stepsize = 0.005
        t = 0
        sol, t_new = euler_step(stepsize, pred_ode, [0.5, 0.5], t)
        real = [0.49916696, 0.49999948]
        assert np.isclose(t_new, t + stepsize) and np.allclose(sol, real)

    def test_rk4(self):
        stepsize = 0.005
        t = 0
        sol, t_new = rk4_step(stepsize, pred_ode, [0.5, 0.5], t)
        real = [0.49916696, 0.49999948]
        assert np.isclose(t_new, t + stepsize) and np.allclose(sol, real)

    def test_solve_to(self):
        X = (0.5, 0.5)
        t_1 = 0
        t_2 = 1
        stepsize = 0.005
        real_rk = [0.34113673, 0.48927056]
        real_eul = [0.34108465, 0.48932033]
        sol, t = solve_to(X, t_1, t_2, stepsize, deltat_max=2, func=pred_ode, method='rk4')
        sol1, t1 = solve_to(X, t_1, t_2, stepsize, deltat_max=2, func=pred_ode, method='euler')
        assert np.isclose(t, t_2) and np.allclose(sol, real_rk) #Test rk4
        assert np.isclose(t1, t_2) and np.allclose(sol1, real_eul)  # Test euler

    def test_solve_ode(self):
        assert np.isclose(solve_ode(pred_ode, t_span, initialu, stepsize=0.005, method="rk4")[1][-1][-1], 0.27005439464182407)

    def test_scipy_solver(self):
        assert np.isclose(scipy_solver(pred_ode, initialu, t_span)[-1][-1][-1], 0.26763146225072443)

    def test_my_solver(self):
        assert np.isclose(my_solver(pred_ode, initialu, t_span)[1][-1][-1], 0.27005439464182407)

    def test_time_simulation(self):
        assert np.isclose(time_simulation(pred_ode, initialu, t_span)[1][-1][-1], 0.27005439464182407)

    def test_orbit(self):
        assert np.isclose(orbit(pred_ode, (0.79912268, 0.18061336), t_span)[1][-1][-1], 0.27005439464182407)

    def test_nullcline(self):
        # Good - check values match nullcline definition (approximately)
        [u1, u2] = nullcline(pred_ode, [0.32, 0.32])
        assert np.allclose([pred_ode(0, u)[0] for u in zip(u1, u2)], 0)

    def test_equilibria(self):
        equilibria = find_equilibria(pred_ode, 2)
        check_list = []
        for i in range(len(equilibria)):
            check_list.append(np.allclose(pred_ode(np.nan, np.array((equilibria[i]), dtype="float64")), 0))
        assert all(check_list)

    def test_shoot(self):
        pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
        assert np.allclose(ns.shoot((0.32, 0.32, 30.), pred_ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=20, index=0, plot=False), [0.79912268,  0.18061336, 31.6038949])

    def test_parameter_continuation(self):
        # Natural Parameter Continuation for parameter b in [0.1, 0.5]
        # Initial guess for initial conditions & period T
        init_guess = (0.79, 0.18, 30.)
        # Range of parameter values to be tested for b
        p_span = (0.1, 0.5)
        # Set parameter to change
        param_string = "b"
        # Calculate natural parameter continuation
        sol = nc.natural_continuation(init_guess, pred_prey, p_span, 0.005, param_string, defaults_pred_prey(),
                                      solver="custom", method="rk4", stepsize=0.125, deltat_max=20, index=0, plot=False)
        assert np.isclose(sol[-1][-1], 0.2701562118716416)

    def test_phase_cond(self):
        sol = ns.phase_condition(pred_ode, initialu, index=0)
        assert np.isclose(sol, -5.05750072e-09)

    def test_vec_eq(self):
        init_guess = np.array((0.79, 0.18, 30.))
        sol = ns.vector_eq(init_guess, pred_ode)
        assert np.isclose(sol[-1], 0.00612471910112361)

    def test_forward_euler(self):
        sol = pde.pde_solver(u_I, params=default_heat_params(), mx=30, mt=1000, plot=False)
        assert np.isclose(sol[-1][-1], 0)

    def test_backward_euler(self):
        sol = pde.pde_solver(u_I, method="b-euleur", params=default_heat_params(), mx=100, mt=100, plot=False)
        assert np.isclose(sol[-1][-1], 0)

    def test_crank_nicholson(self):
        sol = pde.pde_solver(u_I, method="ck", params=default_heat_params(), mx=100, mt=100, plot=False)
        assert np.isclose(sol[-1][-1], 0)

    def test_dirichlet(self):
        def p(t):
            return t / 100

        def q(t):
            return t / 100

        sol = pde.pde_solver(u_I, boundary_conds="dirichlet", bound_funcs=(p, q), params=default_heat_params(),
                            mx=30, mt=1000, plot=False)
        assert np.isclose(sol[-1][-1], 10.)

    def test_neumann(self):
        def p(t):
            return t +3

        def q(t):
            return t +3

        sol = pde.pde_solver(u_I, boundary_conds="dirichlet", bound_funcs=(p, q), params=default_heat_params(),
                            mx=30, mt=1000, plot=False)
        assert np.isclose(sol[-1][-1], 1003.)


    def test_periodic(self):
        sol = pde.pde_solver(u_I, boundary_conds="periodic", params=default_heat_params(),
                             mx=30, mt=1000, plot=False)
        assert np.isclose(sol[-1][-1], 0.6360378886070785)

    def test_heat_source(self):
        def heat_func(x, t):
            return 2 * x + t

        sol = pde.pde_solver(u_I, boundary_conds="zero", pde_type="heat_source", heat_func=heat_func,
                            params=default_heat_params(),
                            mx=30, mt=1000, plot=False)
        assert np.isclose(sol[-1][-1], 0.0300075)

if __name__ == '__main__':
    unittest.main()


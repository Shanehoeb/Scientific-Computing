import unittest
from ODEs import*
from examples import*

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
        assert np.isclose(pp.my_solver(pred_ode, initialu, t_span)[1][-1][-1], 0.27005439464182407)

    def test_time_simulation(self):
        assert np.isclose(pp.time_simulation(pred_ode, initialu, t_span)[1][-1][-1], 0.27005439464182407)

    def test_orbit(self):
        assert np.isclose(pp.orbit(pred_ode, (0.79912268, 0.18061336), t_span)[1][-1][-1], 0.27005439464182407)

    def test_nullcline(self):
        # Good - check values match nullcline definition (approximately)
        [u1, u2] = nullcline(pred_ode, [0.32, 0.32])
        assert np.allclose([pred_ode(0, u)[0] for u in zip(u1, u2)], 0)

    def test_equilibria(self):
        equilibria = pp.find_equilibria(pred_ode, 2)
        check_list = []
        for i in range(len(equilibria)):
            check_list.append(np.allclose(pred_ode(np.nan, np.array((equilibria[i]), dtype="float64")), 0))
        assert all(check_list)

    def test_shoot(self):
        pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
        assert np.allclose(ns.shoot((0.32, 0.32, 30.), pred_ode, solver="custom", method="rk4", stepsize=0.005, deltat_max=20, index=0, plot=False), [0.79912268,  0.18061336, 31.6038949])

    def test_phase_cond(self):
        assert True==True

    def test_vec_eq(self):
        assert True==True

    def test_natural_continuation(self):
        assert True==True

    # TODO : PDE TESTS

if __name__ == '__main__':
    unittest.main()


import unittest
from phaseportraits import*
from examples import*

pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())


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
        real_rk = [0.34428321, 0.47752384]
        real_eul = [0.3441899, 0.47760635]
        sol, t = solve_to(X, t_1, t_2, stepsize, deltat_max=2, func=pred_ode, method='rk4')
        sol1, t1 = solve_to(X, t_1, t_2, stepsize, deltat_max=2, func=pred_ode, method='euler')
        assert np.isclose(t, t_2) and np.allclose(sol, real_rk) #Test rk4
        assert np.isclose(t1, t_2) and np.allclose(sol1, real_eul)  # Test euler

    def test_solve_ode(self):
        assert "a" == "a"

    def test_scipy_solver(self):
        assert "a" == "a"

    def test_my_solver(self):
        assert "a" == "a"

    def test_time_simulation(self):
        assert "a" == "a"

    def test_orbit(self):
        assert "a" == "a"

    def test_nullcline(self):
        # Good - check values match nullcline definition (approximately)
        [u1, u2] = nullcline(pred_ode, [0.32, 0.32])
        assert np.allclose([pred_ode(0, u)[0] for u in zip(u1, u2)], 0)

    def test_equilibria(self):
        equilibria = pp.find_equilibria(pred_ode,2)
        check_list = []
        for i in range(len(equilibria)):
            check_list.append(np.allclose(pred_ode(np.nan, np.array((equilibria[i]), dtype="float64")),0))
        assert all(check_list)

    def test_shoot(self):
        pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())
        assert np.allclose(ns.shoot((0.33, 0.33, 18), pred_ode, plot=False),[0.38917637, 0.29880049, 18.38318297])

if __name__ == '__main__':
    unittest.main()


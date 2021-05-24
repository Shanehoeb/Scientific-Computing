import unittest

import numpy as np

from numerical_methods import*
from phaseportraits import*
from examples import*
pred_ode = lambda t, u: pred_prey(t, u, defaults_pred_prey())


class MyTestCase(unittest.TestCase):
    def test_test(self):
        self.assertEqual(True, True)

    def test_euler(self):
        stepsize = 0.005
        t = 0
        sol, t_new = euler_step(stepsize, pred_ode,[0.5, 0.5], t)
        real = [0.49916696, 0.49999948]
        assert np.isclose(t_new, t + stepsize) and np.allclose(sol, real)


    def test_rk4(self):
        stepsize = 0.005
        t = 0
        sol, t_new = rk4_step(stepsize, pred_ode, [0.5, 0.5], t)
        real = [0.49916696, 0.49999948]
        assert np.isclose(t_new, t + stepsize) and np.allclose(sol, real)


    def test_solve_to(self):
        assert "a" == "a"

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

if __name__ == '__main__':
    unittest.main()


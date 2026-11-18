import unittest

import numpy as np
import pandas as pd

from rebalance.simulate import Strategy, _rebalance, compare, parse_weights, simulate

IDX = pd.period_range("2020-01", periods=4, freq="M")


class SimulateTest(unittest.TestCase):
    def setUp(self):
        # 주식: +10%, -10%, +10%, -10% / 채권: 0%
        self.returns = pd.DataFrame({"주식": [0.1, -0.1, 0.1, -0.1], "채권": [0.0] * 4}, index=IDX)
        self.w = {"주식": 0.5, "채권": 0.5}

    def test_buy_and_hold_matches_manual(self):
        r = simulate(self.returns, self.w, Strategy("hold", "none"), initial_value=100)
        expected = 50 * 1.1 * 0.9 * 1.1 * 0.9 + 50
        self.assertAlmostEqual(r.values.iloc[-1], expected)
        self.assertEqual(r.rebalance_dates, [])

    def test_monthly_rebalance_matches_manual(self):
        r = simulate(self.returns, self.w, Strategy("m", "periodic", period_months=1), initial_value=100)
        expected = 100 * 1.05 * 0.95 * 1.05 * 0.95
        self.assertAlmostEqual(r.values.iloc[-1], expected)
        self.assertEqual(len(r.rebalance_dates), 4)
        np.testing.assert_allclose(r.weights.to_numpy(), 0.5)

    def test_threshold(self):
        r = simulate(self.returns, self.w, Strategy("b", "threshold", band=0.03), initial_value=100)
        # 1개월차 주식 비중 55/105 = 52.4% -> 2.4%p, 밴드 미만
        self.assertNotIn(IDX[0], r.rebalance_dates)
        r2 = simulate(self.returns, self.w, Strategy("b", "threshold", band=0.02), initial_value=100)
        self.assertIn(IDX[0], r2.rebalance_dates)

    def test_rebalance_cost_is_consistent(self):
        holdings = np.array([70.0, 30.0])
        target = np.array([0.5, 0.5])
        new, cost, traded = _rebalance(holdings, target, 0.01)
        np.testing.assert_allclose(new / new.sum(), target)
        self.assertAlmostEqual(new.sum() + cost, 100.0)
        self.assertAlmostEqual(cost, 0.01 * traded)

    def test_zero_cost_rebalance_preserves_value(self):
        new, cost, _ = _rebalance(np.array([70.0, 30.0]), np.array([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(new.sum(), 100.0)
        self.assertEqual(cost, 0.0)

    def test_costs_reduce_value(self):
        s = Strategy("m", "periodic", period_months=1)
        free = simulate(self.returns, self.w, s, initial_value=100).values.iloc[-1]
        paid = simulate(self.returns, self.w, s, initial_value=100, cost_bp=50).values.iloc[-1]
        self.assertLess(paid, free)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            simulate(self.returns, {"주식": 0.6, "채권": 0.6}, Strategy("h", "none"))
        with self.assertRaises(ValueError):
            simulate(self.returns, {"주식": 0.5, "현금": 0.5}, Strategy("h", "none"))

    def test_parse_weights_percent(self):
        self.assertEqual(parse_weights("주식=60,채권=40"), {"주식": 0.6, "채권": 0.4})
        self.assertEqual(parse_weights("주식=0.6,채권=0.4"), {"주식": 0.6, "채권": 0.4})

    def test_compare_table(self):
        table, _ = compare(self.returns, self.w, initial_value=100)
        self.assertIn("최대낙폭(MDD)", table.columns)
        self.assertEqual(len(table), 7)


if __name__ == "__main__":
    unittest.main()

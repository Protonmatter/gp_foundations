import unittest

from gp_foundations.updates import TwoTimescaleUpdater


class TwoTimescaleTests(unittest.TestCase):
    def test_interval_schedule_runs_slow_path_less_often(self) -> None:
        updater = TwoTimescaleUpdater(
            fast_update=lambda state, obs, step: (state or 0) + obs,
            slow_update=lambda state, obs, step: state + 100,
            slow_interval=3,
        )
        for obs in [1, 1, 1, 1, 1, 1]:
            updater.update(obs)
        self.assertEqual(updater.fast_updates, 6)
        self.assertEqual(updater.slow_updates, 2)

    def test_logarithmic_schedule_helper(self) -> None:
        schedule = TwoTimescaleUpdater.logarithmic_schedule(base=2, offset=1)
        self.assertTrue(schedule(1))
        self.assertTrue(schedule(3))
        self.assertFalse(schedule(4))


if __name__ == '__main__':
    unittest.main()

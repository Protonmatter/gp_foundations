import ast
import threading
import unittest
from pathlib import Path

import numpy as np

from gp_foundations.multioutput import CoregionalizationMatrix
from gp_foundations.wifi_research import JointStrategySimulator, ReplayEnvironment, StrategyObservation


ROOT = Path(__file__).resolve().parents[1]
WIFI_RESEARCH = ROOT / 'src' / 'gp_foundations' / 'wifi_research'


class WifiResearchTests(unittest.TestCase):
    def test_replay_environment_synthetic_and_ingest(self) -> None:
        strategy_ids = ('strategy_a', 'strategy_b')
        grid = np.linspace(0.0, 1.0, 5)
        environment = ReplayEnvironment.synthetic(
            strategy_ids,
            grid,
            reward_functions={
                'strategy_a': lambda intensity: 1.0 - (intensity - 0.25) ** 2,
                'strategy_b': lambda intensity: 0.8 - (intensity - 0.75) ** 2,
            },
            rounds_per_strategy=3,
            rng=np.random.default_rng(0),
        )
        simulator = JointStrategySimulator(strategy_ids, grid, window_size=10)
        consumed = simulator.ingest_environment(environment)
        self.assertEqual(consumed, 6)
        self.assertEqual(environment.remaining, 0)
        snapshot = simulator.observations_snapshot()
        self.assertEqual(sum(len(records) for records in snapshot.values()), 6)

    def test_sliding_window_keeps_recent_observations_per_strategy(self) -> None:
        simulator = JointStrategySimulator(('strategy_a', 'strategy_b'), [0.0, 0.5, 1.0], window_size=2)
        simulator.record_observation(StrategyObservation('strategy_a', 0.0, 0.1))
        simulator.record_observation(StrategyObservation('strategy_a', 0.5, 0.2))
        simulator.record_observation(StrategyObservation('strategy_a', 1.0, 0.3))
        simulator.record_observation(StrategyObservation('strategy_b', 0.5, 0.4))
        snapshot = simulator.observations_snapshot()
        self.assertEqual(len(snapshot['strategy_a']), 2)
        self.assertEqual([record.intensity for record in snapshot['strategy_a']], [0.5, 1.0])
        self.assertEqual(len(snapshot['strategy_b']), 1)

    def test_recommendations_stay_within_configured_strategy_ids_and_grid(self) -> None:
        simulator = JointStrategySimulator(('strategy_a', 'strategy_b'), [0.0, 0.5, 1.0], window_size=10)
        simulator.record_many(
            [
                StrategyObservation('strategy_a', 0.0, 0.1),
                StrategyObservation('strategy_a', 0.5, 0.9),
                StrategyObservation('strategy_b', 0.5, 0.4),
                StrategyObservation('strategy_b', 1.0, 0.8),
            ]
        )
        recommendations = simulator.evaluate_recommendations(rng=np.random.default_rng(4))
        self.assertEqual({recommendation.strategy_id for recommendation in recommendations}, {'strategy_a', 'strategy_b'})
        self.assertTrue(all(recommendation.intensity in {0.0, 0.5, 1.0} for recommendation in recommendations))

    def test_cross_strategy_correlation_changes_unobserved_posterior(self) -> None:
        grid = [0.25, 0.5, 0.75]
        correlated = JointStrategySimulator(
            ('strategy_a', 'strategy_b'),
            grid,
            coregionalization=CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.95, 0.2]])),
        )
        independent = JointStrategySimulator(
            ('strategy_a', 'strategy_b'),
            grid,
            coregionalization=CoregionalizationMatrix.identity(2),
        )
        observations = [
            StrategyObservation('strategy_a', 0.5, 0.9),
            StrategyObservation('strategy_a', 0.75, 0.6),
        ]
        correlated.record_many(observations)
        independent.record_many(observations)
        correlated_posterior = correlated.posterior_for_strategy('strategy_b')
        independent_posterior = independent.posterior_for_strategy('strategy_b')
        self.assertGreater(np.max(correlated_posterior.mean), np.max(independent_posterior.mean) + 0.1)

    def test_cost_penalty_changes_strategy_ranking(self) -> None:
        grid = [0.0, 0.5, 1.0]
        simulator = JointStrategySimulator(('strategy_a', 'strategy_b'), grid, noise=1e-6, window_size=20)
        observations = []
        for _ in range(4):
            for intensity, reward_a, reward_b in [(0.0, 0.7, 0.5), (0.5, 1.0, 0.8), (1.0, 0.6, 0.4)]:
                observations.append(StrategyObservation('strategy_a', intensity, reward_a, cost=0.9))
                observations.append(StrategyObservation('strategy_b', intensity, reward_b, cost=0.1))
        simulator.record_many(observations)
        baseline = simulator.recommend(rng=np.random.default_rng(3), cost_penalty=0.0)
        penalized = simulator.recommend(
            rng=np.random.default_rng(3),
            cost_penalty=1.0,
            strategy_costs={'strategy_a': 0.9, 'strategy_b': 0.1},
        )
        self.assertEqual(baseline.strategy_id, 'strategy_a')
        self.assertEqual(penalized.strategy_id, 'strategy_b')

    def test_slow_update_callback_runs_on_schedule(self) -> None:
        steps = []

        def hyperparameter_update(histories, current, step):
            steps.append(step)
            return {'noise': current['noise'] + 1e-4}

        simulator = JointStrategySimulator(
            ('strategy_a', 'strategy_b'),
            [0.0, 0.5, 1.0],
            slow_interval=2,
            hyperparameter_update=hyperparameter_update,
        )
        simulator.record_many(
            [
                StrategyObservation('strategy_a', 0.0, 0.2),
                StrategyObservation('strategy_b', 0.5, 0.4),
                StrategyObservation('strategy_a', 1.0, 0.1),
                StrategyObservation('strategy_b', 1.0, 0.3),
                StrategyObservation('strategy_a', 0.5, 0.9),
            ]
        )
        self.assertEqual(steps, [2, 4])
        self.assertEqual(simulator.updater.slow_updates, 2)
        self.assertGreater(simulator.hyperparameters_snapshot()['noise'], 1e-6)

    def test_near_singular_joint_evaluation_remains_finite(self) -> None:
        coregionalization = CoregionalizationMatrix.from_factor(np.array([[1.0, 0.0], [0.999999, 1e-6]]))
        simulator = JointStrategySimulator(
            ('strategy_a', 'strategy_b'),
            [0.25, 0.5, 0.75],
            coregionalization=coregionalization,
            noise=1e-12,
        )
        simulator.record_many(
            [
                StrategyObservation('strategy_a', 0.5, 1.0),
                StrategyObservation('strategy_a', 0.5, 1.0),
                StrategyObservation('strategy_b', 0.5, 0.9),
                StrategyObservation('strategy_b', 0.5, 0.9),
            ]
        )
        evaluation = simulator.joint_grid_evaluation(rng=np.random.default_rng(5))
        self.assertTrue(np.all(np.isfinite(evaluation.posterior.mean)))
        self.assertTrue(np.all(np.isfinite(evaluation.posterior.variance)))
        self.assertTrue(np.all(np.isfinite(evaluation.sample)))

    def test_concurrent_updates_and_snapshot_reads_do_not_corrupt_state(self) -> None:
        simulator = JointStrategySimulator(('strategy_a', 'strategy_b'), np.linspace(0.0, 1.0, 5), window_size=20)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for index in range(40):
                    strategy_id = 'strategy_a' if index % 2 == 0 else 'strategy_b'
                    intensity = float((index % 5) / 4.0)
                    reward = 1.0 - abs(intensity - 0.5)
                    simulator.record_observation(StrategyObservation(strategy_id, intensity, reward, cost=0.2))
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=writer)
        thread.start()
        for _ in range(20):
            try:
                simulator.evaluate_recommendations(rng=np.random.default_rng(7), cost_penalty=0.2)
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
        thread.join()
        self.assertEqual(errors, [])
        snapshot = simulator.observations_snapshot()
        self.assertTrue(all(len(records) <= 20 for records in snapshot.values()))

    def test_wifi_research_module_has_no_live_network_or_process_imports(self) -> None:
        forbidden_modules = {'socket', 'subprocess', 'scapy', 'pyroute2', 'netifaces', 'pyric'}
        for python_file in WIFI_RESEARCH.glob('*.py'):
            with self.subTest(file=python_file.name):
                tree = ast.parse(python_file.read_text())
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name.split('.')[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module is not None:
                        imported.add(node.module.split('.')[0])
                self.assertTrue(forbidden_modules.isdisjoint(imported))


if __name__ == '__main__':
    unittest.main()

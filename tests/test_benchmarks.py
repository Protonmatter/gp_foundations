import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / 'benchmarks'


class BenchmarkScriptTests(unittest.TestCase):
    def _run(self, script_name: str, *args: str) -> dict:
        script_path = BENCHMARKS / script_name
        result = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_regret_benchmark_outputs_summary(self) -> None:
        payload = self._run('benchmark_regret.py', '--trials', '2', '--grid-size', '7', '--optimization-maxiter', '8')
        self.assertEqual(payload['metric'], 'regret')
        self.assertIn('fixed', payload)
        self.assertIn('learned', payload)

    def test_calibration_benchmark_outputs_summary(self) -> None:
        payload = self._run(
            'benchmark_calibration.py',
            '--trials', '2',
            '--n-train', '6',
            '--n-test', '8',
            '--optimization-maxiter', '10',
        )
        self.assertEqual(payload['metric'], 'calibration')
        self.assertIn('coverage', payload['fixed'])
        self.assertIn('calibration_error', payload['learned'])

    def test_runtime_benchmark_outputs_summary(self) -> None:
        payload = self._run(
            'benchmark_runtime.py',
            '--trials', '2',
            '--n-train', '6',
            '--n-test', '8',
            '--grid-size', '7',
            '--optimization-maxiter', '8',
        )
        self.assertEqual(payload['metric'], 'runtime_seconds')
        self.assertIn('gp_fixed', payload)
        self.assertIn('simulator_learned', payload)


if __name__ == '__main__':
    unittest.main()

# Offline WiFi Research Workflow

`gp_foundations.wifi_research` is an offline experimentation layer for replay and simulation. It does not interact with live interfaces or network stacks.

## Workflow

1. Define neutral strategy labels such as `strategy_a`, `strategy_b`, and `strategy_c`.
2. Build a bounded intensity grid in `[0, 1]`.
3. Create a `ReplayEnvironment` from historical observations or synthetic reward functions.
4. Feed observations into `JointStrategySimulator`.
5. Call `evaluate_recommendations()` to rank strategy and intensity pairs from one coherent joint sample.
6. Call `posterior_for_strategy()` when you want per-strategy posterior summaries across the grid.
7. Use the built-in `TwoTimescaleUpdater` integration when you need slower hyperparameter refreshes than observation ingestion.

## Minimal example

```python
import numpy as np

from gp_foundations.multioutput import CoregionalizationMatrix
from gp_foundations.wifi_research import JointStrategySimulator, ReplayEnvironment

strategy_ids = ("strategy_a", "strategy_b")
grid = np.linspace(0.0, 1.0, 21)

environment = ReplayEnvironment.synthetic(
    strategy_ids,
    grid,
    reward_functions={
        "strategy_a": lambda intensity: 0.9 - 3.0 * (intensity - 0.35) ** 2,
        "strategy_b": lambda intensity: 0.8 - 2.2 * (intensity - 0.7) ** 2,
    },
    rounds_per_strategy=8,
    noise_std=0.02,
    rng=np.random.default_rng(7),
)

simulator = JointStrategySimulator(
    strategy_ids,
    grid,
    coregionalization=CoregionalizationMatrix.from_factor(
        np.array([[1.0, 0.0], [0.6, 0.4]])
    ),
    slow_interval=4,
)
simulator.ingest_environment(environment)
recommendation = simulator.recommend(rng=np.random.default_rng(11), cost_penalty=0.2)
```

## Safety boundary

This layer is intentionally limited to offline or replay data. The module does not import raw socket APIs, `scapy`, `subprocess`, or interface-control libraries, and the tests enforce that boundary.

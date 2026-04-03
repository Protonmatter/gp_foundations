from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..kernels import MaternKernel
from ..multioutput import (
    CoregionalizationMatrix,
    IntrinsicCoregionalizedGP,
    JointGridEvaluation,
    OutputPosterior,
)
from ..runtime import SnapshotStore
from ..updates import TwoTimescaleUpdater


@dataclass(frozen=True)
class StrategyObservation:
    strategy_id: str
    intensity: float
    reward: float
    cost: float = 0.0
    context: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id must be non-empty")
        if not (0.0 <= float(self.intensity) <= 1.0):
            raise ValueError("intensity must be in [0, 1]")
        if not np.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if not np.isfinite(float(self.cost)):
            raise ValueError("cost must be finite")
        if float(self.cost) < 0.0:
            raise ValueError("cost must be non-negative")
        if self.context is not None:
            context = np.asarray(self.context, dtype=float).reshape(-1)
            if not np.all(np.isfinite(context)):
                raise ValueError("context must contain only finite values")
            object.__setattr__(self, "context", context)


@dataclass(frozen=True)
class StrategyRecommendation:
    strategy_id: str
    intensity: float
    sampled_score: float
    posterior_mean: float
    posterior_variance: float
    penalized_score: float


class ReplayEnvironment:
    def __init__(
        self,
        strategy_ids: Sequence[str],
        observations: Iterable[StrategyObservation],
        intensity_grid: Sequence[float],
    ):
        self.strategy_ids = _validate_strategy_ids(strategy_ids)
        self.intensity_grid = _validate_intensity_grid(intensity_grid)
        self._observations = [self._validate_observation(observation) for observation in observations]
        self._cursor = 0

    @classmethod
    def synthetic(
        cls,
        strategy_ids: Sequence[str],
        intensity_grid: Sequence[float],
        reward_functions: Mapping[str, Callable[[float], float]],
        *,
        rounds_per_strategy: int = 6,
        noise_std: float = 0.0,
        cost_functions: Mapping[str, Callable[[float], float]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> "ReplayEnvironment":
        strategy_list = _validate_strategy_ids(strategy_ids)
        grid = _validate_intensity_grid(intensity_grid)
        generator = rng or np.random.default_rng()
        observations: list[StrategyObservation] = []
        for _ in range(rounds_per_strategy):
            for strategy_id in strategy_list:
                if strategy_id not in reward_functions:
                    raise KeyError(f"missing reward function for strategy {strategy_id!r}")
                intensity = float(generator.choice(grid))
                reward = float(reward_functions[strategy_id](intensity))
                if noise_std > 0.0:
                    reward += float(generator.normal(0.0, noise_std))
                cost = 0.0
                if cost_functions is not None and strategy_id in cost_functions:
                    cost = float(cost_functions[strategy_id](intensity))
                observations.append(
                    StrategyObservation(
                        strategy_id=strategy_id,
                        intensity=intensity,
                        reward=reward,
                        cost=max(cost, 0.0),
                    )
                )
        return cls(strategy_list, observations, grid)

    def _validate_observation(self, observation: StrategyObservation) -> StrategyObservation:
        if observation.strategy_id not in self.strategy_ids:
            raise ValueError(f"unknown strategy_id: {observation.strategy_id}")
        return observation

    def reset(self) -> None:
        self._cursor = 0

    @property
    def remaining(self) -> int:
        return len(self._observations) - self._cursor

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._observations)

    def step(self) -> StrategyObservation:
        if self.done:
            raise StopIteration("replay is exhausted")
        observation = self._observations[self._cursor]
        self._cursor += 1
        return observation

    def __iter__(self) -> Iterator[StrategyObservation]:
        self.reset()
        while not self.done:
            yield self.step()

    def observations(self) -> list[StrategyObservation]:
        return list(self._observations)


class JointStrategySimulator:
    def __init__(
        self,
        strategy_ids: Sequence[str],
        intensity_grid: Sequence[float],
        *,
        kernel: object | None = None,
        coregionalization: CoregionalizationMatrix | None = None,
        noise: float = 1e-6,
        window_size: int = 100,
        slow_interval: int = 10,
        warmup: int = 0,
        enable_built_in_optimization: bool = False,
        optimization_maxiter: int = 50,
        optimize_noise: bool = True,
        hyperparameter_update: Callable[[dict[str, list[StrategyObservation]], dict[str, Any], int], Mapping[str, Any] | CoregionalizationMatrix | None] | None = None,
    ):
        self.strategy_ids = _validate_strategy_ids(strategy_ids)
        self.strategy_index = {strategy_id: idx for idx, strategy_id in enumerate(self.strategy_ids)}
        self.intensity_grid = _validate_intensity_grid(intensity_grid)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        if optimization_maxiter <= 0:
            raise ValueError("optimization_maxiter must be positive")
        self.window_size = int(window_size)
        self.enable_built_in_optimization = bool(enable_built_in_optimization)
        self.optimization_maxiter = int(optimization_maxiter)
        self.optimize_noise = bool(optimize_noise)
        self._data_store: SnapshotStore[dict[str, list[StrategyObservation]]] = SnapshotStore(
            {strategy_id: [] for strategy_id in self.strategy_ids}
        )
        self._hyperparameter_store: SnapshotStore[dict[str, Any]] = SnapshotStore(
            {
                "kernel": kernel if kernel is not None else MaternKernel(length_scale=0.2),
                "coregionalization": coregionalization or CoregionalizationMatrix.identity(len(self.strategy_ids)),
                "noise": float(noise),
            }
        )
        self._hyperparameter_update = hyperparameter_update
        self._updater = TwoTimescaleUpdater(
            fast_update=self._fast_update,
            slow_update=self._slow_update,
            initial_state={"steps": 0},
            slow_interval=slow_interval,
            warmup=warmup,
        )

    @property
    def updater(self) -> TwoTimescaleUpdater:
        return self._updater

    def observations_snapshot(self) -> dict[str, list[StrategyObservation]]:
        return self._data_store.snapshot()

    def hyperparameters_snapshot(self) -> dict[str, Any]:
        return self._hyperparameter_store.snapshot()

    def record_observation(self, observation: StrategyObservation) -> dict[str, int]:
        observation = self._validate_observation(observation)
        return self._updater.update(observation)

    def record_many(self, observations: Iterable[StrategyObservation]) -> None:
        for observation in observations:
            self.record_observation(observation)

    def ingest_environment(self, environment: ReplayEnvironment, limit: int | None = None) -> int:
        steps = 0
        for observation in environment:
            if limit is not None and steps >= limit:
                break
            self.record_observation(observation)
            steps += 1
        return steps

    def joint_grid_evaluation(self, rng: np.random.Generator | None = None) -> JointGridEvaluation:
        histories = self.observations_snapshot()
        hyperparameters = self.hyperparameters_snapshot()
        model = self._build_model(histories, hyperparameters)
        return model.evaluate_grid(self.intensity_grid[:, None], rng=rng)

    def posterior_for_strategy(self, strategy_id: str, return_covariance: bool = False) -> OutputPosterior:
        strategy_index = self._lookup_strategy_index(strategy_id)
        histories = self.observations_snapshot()
        hyperparameters = self.hyperparameters_snapshot()
        model = self._build_model(histories, hyperparameters)
        return model.posterior_for_output(
            self.intensity_grid[:, None],
            strategy_index,
            return_covariance=return_covariance,
        )

    def evaluate_recommendations(
        self,
        *,
        rng: np.random.Generator | None = None,
        cost_penalty: float = 0.0,
        strategy_costs: Mapping[str, float] | None = None,
    ) -> list[StrategyRecommendation]:
        if cost_penalty < 0.0:
            raise ValueError("cost_penalty must be non-negative")
        histories = self.observations_snapshot()
        evaluation = self.joint_grid_evaluation(rng=rng)
        resolved_costs = self._resolve_strategy_costs(histories, strategy_costs)
        recommendations: list[StrategyRecommendation] = []
        for output_index, strategy_id in enumerate(self.strategy_ids):
            sample = evaluation.sample[:, output_index]
            penalty = float(cost_penalty) * resolved_costs[strategy_id]
            penalized_sample = sample - penalty
            best_index = int(np.argmax(penalized_sample))
            recommendations.append(
                StrategyRecommendation(
                    strategy_id=strategy_id,
                    intensity=float(self.intensity_grid[best_index]),
                    sampled_score=float(sample[best_index]),
                    posterior_mean=float(evaluation.posterior.mean[best_index, output_index]),
                    posterior_variance=float(evaluation.posterior.variance[best_index, output_index]),
                    penalized_score=float(penalized_sample[best_index]),
                )
            )
        return sorted(recommendations, key=lambda recommendation: recommendation.penalized_score, reverse=True)

    def recommend(
        self,
        *,
        rng: np.random.Generator | None = None,
        cost_penalty: float = 0.0,
        strategy_costs: Mapping[str, float] | None = None,
    ) -> StrategyRecommendation:
        return self.evaluate_recommendations(
            rng=rng,
            cost_penalty=cost_penalty,
            strategy_costs=strategy_costs,
        )[0]

    def _validate_observation(self, observation: StrategyObservation) -> StrategyObservation:
        if observation.strategy_id not in self.strategy_index:
            raise ValueError(f"unknown strategy_id: {observation.strategy_id}")
        return observation

    def _lookup_strategy_index(self, strategy_id: str) -> int:
        try:
            return self.strategy_index[strategy_id]
        except KeyError as exc:
            raise ValueError(f"unknown strategy_id: {strategy_id}") from exc

    def _fast_update(self, state: dict[str, int] | None, observation: StrategyObservation, step: int) -> dict[str, int]:
        self._data_store.mutate(lambda history: self._append_observation(history, observation))
        return {"steps": step}

    def _slow_update(self, state: dict[str, int], observation: StrategyObservation, step: int) -> dict[str, int]:
        histories = self.observations_snapshot()
        current = self.hyperparameters_snapshot()
        if self._hyperparameter_update is not None:
            updated = self._hyperparameter_update(histories, current, step)
        elif self.enable_built_in_optimization:
            updated = self._built_in_hyperparameter_update(histories, current)
        else:
            return state
        if updated is not None:
            self._hyperparameter_store.replace(self._normalize_hyperparameters(updated, current))
        return state

    def _append_observation(
        self,
        history: dict[str, list[StrategyObservation]],
        observation: StrategyObservation,
    ) -> dict[str, list[StrategyObservation]]:
        records = list(history[observation.strategy_id])
        records.append(observation)
        history[observation.strategy_id] = records[-self.window_size :]
        return history

    def _resolve_strategy_costs(
        self,
        histories: dict[str, list[StrategyObservation]],
        strategy_costs: Mapping[str, float] | None,
    ) -> dict[str, float]:
        resolved: dict[str, float] = {}
        for strategy_id in self.strategy_ids:
            if strategy_costs is not None and strategy_id in strategy_costs:
                cost = float(strategy_costs[strategy_id])
                if cost < 0.0:
                    raise ValueError("strategy_costs must be non-negative")
                resolved[strategy_id] = cost
                continue
            observations = histories[strategy_id]
            if observations:
                resolved[strategy_id] = float(np.mean([observation.cost for observation in observations]))
            else:
                resolved[strategy_id] = 0.0
        return resolved

    def _build_model(
        self,
        histories: dict[str, list[StrategyObservation]],
        hyperparameters: dict[str, Any],
    ) -> IntrinsicCoregionalizedGP:
        kernel = hyperparameters["kernel"]
        coregionalization = hyperparameters["coregionalization"]
        noise = float(hyperparameters["noise"])
        model = IntrinsicCoregionalizedGP(kernel, coregionalization, noise=noise)

        X_values: list[list[float]] = []
        y_values: list[float] = []
        output_indices: list[int] = []
        for strategy_id in self.strategy_ids:
            for observation in histories[strategy_id]:
                X_values.append([observation.intensity])
                y_values.append(observation.reward)
                output_indices.append(self.strategy_index[strategy_id])

        if X_values:
            model.fit_observations(
                np.asarray(X_values, dtype=float),
                np.asarray(y_values, dtype=float),
                np.asarray(output_indices, dtype=int),
            )
        return model

    def _built_in_hyperparameter_update(
        self,
        histories: dict[str, list[StrategyObservation]],
        current: dict[str, Any],
    ) -> Mapping[str, Any] | None:
        total_observations = sum(len(records) for records in histories.values())
        if total_observations < max(3, len(self.strategy_ids)):
            return None

        model = self._build_model(histories, current)
        result = model.optimize_hyperparameters(
            optimize_noise=self.optimize_noise,
            maxiter=self.optimization_maxiter,
        )
        if not np.isfinite(result.objective):
            return None
        return {
            "kernel": model.kernel,
            "coregionalization": model.coregionalization,
            "noise": model.noise,
        }

    def _normalize_hyperparameters(
        self,
        updated: Mapping[str, Any] | CoregionalizationMatrix,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(updated, CoregionalizationMatrix):
            merged = {**current, "coregionalization": updated}
        elif isinstance(updated, Mapping):
            merged = {**current, **dict(updated)}
        else:
            raise TypeError("hyperparameter_update must return a mapping, CoregionalizationMatrix, or None")

        kernel = merged.get("kernel")
        coregionalization = merged.get("coregionalization")
        noise = float(merged.get("noise", current["noise"]))

        if kernel is None:
            raise ValueError("kernel must be provided")
        if not isinstance(coregionalization, CoregionalizationMatrix):
            raise TypeError("coregionalization must be a CoregionalizationMatrix")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")

        return {
            "kernel": kernel,
            "coregionalization": coregionalization,
            "noise": noise,
        }


def _validate_strategy_ids(strategy_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(strategy_ids)
    if not values:
        raise ValueError("strategy_ids must be non-empty")
    if any(not strategy_id for strategy_id in values):
        raise ValueError("strategy_ids must not contain empty values")
    if len(set(values)) != len(values):
        raise ValueError("strategy_ids must be unique")
    return values


def _validate_intensity_grid(intensity_grid: Sequence[float]) -> np.ndarray:
    grid = np.asarray(intensity_grid, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("intensity_grid must be non-empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("intensity_grid must be finite")
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("intensity_grid must stay within [0, 1]")
    return np.unique(np.sort(grid))

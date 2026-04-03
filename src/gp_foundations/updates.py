from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class UpdateRecord:
    step: int
    observation: Any
    ran_slow_update: bool


class TwoTimescaleUpdater:
    def __init__(
        self,
        fast_update: Callable[[Any, Any, int], Any],
        slow_update: Callable[[Any, Any, int], Any],
        *,
        initial_state: Any = None,
        slow_interval: int = 10,
        warmup: int = 0,
        schedule: Callable[[int], bool] | None = None,
    ):
        if slow_interval <= 0:
            raise ValueError("slow_interval must be positive")
        self.fast_update = fast_update
        self.slow_update = slow_update
        self.state = initial_state
        self.slow_interval = int(slow_interval)
        self.warmup = int(warmup)
        self.schedule = schedule
        self.step_count = 0
        self.fast_updates = 0
        self.slow_updates = 0
        self.history: list[UpdateRecord] = []

    @staticmethod
    def logarithmic_schedule(base: int = 2, offset: int = 1) -> Callable[[int], bool]:
        if base <= 1:
            raise ValueError("base must be greater than 1")
        if offset < 0:
            raise ValueError("offset must be non-negative")

        def _schedule(step: int) -> bool:
            value = step + offset
            while value % base == 0:
                value //= base
            return value == 1

        return _schedule

    def should_run_slow_update(self, step: int | None = None) -> bool:
        current_step = self.step_count if step is None else step
        if self.schedule is not None:
            return bool(self.schedule(current_step))
        if current_step <= self.warmup:
            return False
        return (current_step - self.warmup) % self.slow_interval == 0

    def update(self, observation: Any) -> Any:
        self.step_count += 1
        self.state = self.fast_update(self.state, observation, self.step_count)
        self.fast_updates += 1

        ran_slow = self.should_run_slow_update(self.step_count)
        if ran_slow:
            self.state = self.slow_update(self.state, observation, self.step_count)
            self.slow_updates += 1

        self.history.append(UpdateRecord(step=self.step_count, observation=observation, ran_slow_update=ran_slow))
        return self.state

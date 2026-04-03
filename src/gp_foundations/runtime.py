from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar


T = TypeVar("T")
_SENTINEL = object()


class QueueClosed(RuntimeError):
    pass


class SnapshotStore(Generic[T]):
    def __init__(self, initial_value: T):
        self._value = copy.deepcopy(initial_value)
        self._lock = threading.RLock()

    def replace(self, value: T) -> None:
        with self._lock:
            self._value = copy.deepcopy(value)

    def mutate(self, updater: Callable[[T], T]) -> T:
        with self._lock:
            self._value = copy.deepcopy(updater(copy.deepcopy(self._value)))
            return copy.deepcopy(self._value)

    def snapshot(self) -> T:
        with self._lock:
            return copy.deepcopy(self._value)


@dataclass
class WorkerSignal:
    reason: str | None = None

    def __post_init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()

    def set(self, reason: str | None = None) -> None:
        with self._lock:
            if reason is not None:
                self.reason = reason
            self._event.set()

    def clear(self) -> None:
        with self._lock:
            self.reason = None
            self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_set(self) -> None:
        if self._event.is_set():
            raise RuntimeError(self.reason or "worker signal set")


class ProducerConsumerQueue(Generic[T]):
    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._closed = False
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def put(self, item: T, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                raise QueueClosed("queue is closed")
        self._queue.put(item, timeout=timeout)

    def get(self, timeout: float | None = None) -> T:
        item = self._queue.get(timeout=timeout)
        if item is _SENTINEL:
            self._queue.put(_SENTINEL)
            raise QueueClosed("queue is closed")
        return item

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_SENTINEL)

    def consume(self, timeout: float | None = 0.1) -> Iterator[T]:
        while True:
            try:
                yield self.get(timeout=timeout)
            except queue.Empty:
                if self.closed:
                    break
                continue
            except QueueClosed:
                break

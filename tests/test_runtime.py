import threading
import unittest

from gp_foundations.runtime import ProducerConsumerQueue, QueueClosed, SnapshotStore, WorkerSignal


class RuntimeTests(unittest.TestCase):
    def test_snapshot_store_is_thread_safe(self) -> None:
        store = SnapshotStore({'count': 0})

        def worker() -> None:
            for _ in range(100):
                store.mutate(lambda state: {'count': state['count'] + 1})

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(store.snapshot()['count'], 400)

    def test_worker_signal_raises_when_set(self) -> None:
        signal = WorkerSignal()
        signal.set('stop requested')
        with self.assertRaises(RuntimeError):
            signal.raise_if_set()

    def test_producer_consumer_queue_closes_cleanly(self) -> None:
        queue = ProducerConsumerQueue()
        queue.put(1)
        queue.put(2)
        self.assertEqual(queue.get(), 1)
        self.assertEqual(queue.get(), 2)
        queue.close()
        with self.assertRaises(QueueClosed):
            queue.get(timeout=0.01)
        with self.assertRaises(QueueClosed):
            queue.put(3)


if __name__ == '__main__':
    unittest.main()

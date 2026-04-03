# Concurrency Design Patterns for Scientific Computing

Source PDF: `references/pdfs/07_concurrency_design_patterns.pdf`

## Repository mapping

- Package module: `gp_foundations.runtime`
- Notebook: `notebooks/06_concurrency_patterns.ipynb`

## Core ideas

The PDF focuses on lock discipline, snapshotting, producer-consumer pipelines, and clean thread lifecycle control.

## Implemented surface

- `SnapshotStore`: copy-based snapshots for shared state with minimal lock hold time
- `WorkerSignal`: event-style shutdown or interruption signal with a reason string
- `ProducerConsumerQueue`: closeable queue abstraction for single-producer or small-pipeline experiments

The runtime helpers are deliberately small and testable rather than framework-heavy.

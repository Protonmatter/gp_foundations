# WiFi Research Source Mapping

This repository folds several legacy WiFi MOGP artifacts into a safe offline research layer inside `gp_foundations`. The legacy materials were treated as inputs, not as source of truth.

## Source inputs

- `MATHEMATICAL_IMPLEMENTATION_SUMMARY.md`
- `WiFi Adversarial Framework v12.2.py`
- `mogp_core.py`
- `INTEGRATION_GUIDE.md`
- `output.txt`

## What was adopted

- Joint Thompson sampling over a bounded intensity grid
- Coregionalization-based multi-output GP structure
- Sliding windows for non-stationary replay experiments
- Snapshot-oriented separation between observation state and hyperparameter state
- Numerical safeguards around jitter, symmetry, and near-singular covariance handling

## What was corrected

- The legacy `mogp_core.py` sketch was not ported directly. Its intent now lives in the existing `IntrinsicCoregionalizedGP` implementation and the new `wifi_research` package.
- The integration narrative now matches the repository layout. There is no `/usr/local/bin/`, `/etc/wifi_adversarial/`, or installer flow in this repo.
- `output.txt` is treated as a generated status note, not implementation evidence.

## What was rejected

- Attack-arm enums and offensive naming
- Stealth and evasion scoring semantics
- Packet generation, injection, jamming, or target interaction
- Host deployment instructions or operational runtime claims

## Resulting repo surfaces

- `src/gp_foundations/multioutput.py`: additive helpers for joint grid evaluation and per-output posterior summaries
- `src/gp_foundations/wifi_research/`: offline replay environment and joint strategy simulator
- `notebooks/07_wifi_research_simulation.ipynb`: safe end-to-end example based on synthetic replay data

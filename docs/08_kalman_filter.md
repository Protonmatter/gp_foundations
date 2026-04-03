# The Kalman Filter

Source PDF: `references/pdfs/08_kalman_filter.pdf`

## Repository mapping

- Package module: `gp_foundations.filters`
- Notebook: `notebooks/05_kalman_and_chaos.ipynb`

## Core ideas

The Kalman filter is the linear-Gaussian state estimator companion to the GP material. The PDF emphasizes prediction, correction, and the role of the Kalman gain as a trust-balancing term.

## Implemented surface

- `KalmanFilter.predict(control=None)`
- `KalmanFilter.update(measurement)`
- `KalmanFilter.step(measurement, control=None)`

The current class supports dense linear state-space models and uses the Joseph covariance update for better numerical behavior.

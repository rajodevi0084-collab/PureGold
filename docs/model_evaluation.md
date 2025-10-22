# Model Evaluation Framework

This document outlines the quantitative objectives and evaluation
procedures used for the predictive and reinforcement-learning (RL)
modeling streams.

## Core Datasets and Splits

- **Data cadence:** Daily bars with open, high, low, close, and volume
  sourced from the curated parquet feeds under `data/predictive/`.
- **Feature interfaces:** Both predictive and RL trainers rely on the
  shared configuration at `models/predictive/config.yaml`, ensuring
  identical feature engineering, target definitions, and split
  semantics.
- **Holdout logic:** Production candidates must pass both the
  `production_candidate` and `recent_market` splits defined in the YAML
  config. In addition, rolling cross-validation windows (1-year
  training / 3-month validation and 2-year training / 6-month
  validation) guard against regime drift.

## Predictive Modeling KPIs

The baseline gradient boosting models report the following metrics:

- **Root Mean Squared Error (RMSE):** Emphasises the magnitude of
  prediction errors in basis points of future returns. Targets below
  25–35 bps are considered competitive for liquid large-cap equities.
- **Mean Absolute Error (MAE):** Complements RMSE by capturing median
  error magnitudes; values under 20–25 bps indicate stable behaviour.
- **Directional Accuracy:** Share of validation samples where the sign
  of the prediction matches the realised return. Sustainable
  outperformance typically lies in the 52–58% range depending on asset
  class; anything above 60% warrants heightened scrutiny.
- **Strategy Sharpe (ex ante):** Computes the risk-adjusted return of a
  naive sign-following strategy using the model’s predictions.
  Consistent Sharpe ratios between 0.8 and 1.2 on validation data are
  realistic prior to transaction costs.
- **Average Strategy Return & Volatility:** Provide visibility into the
  raw expectancy and drawdown potential when deploying directional
  signals without portfolio optimisation.

All metrics are logged per train/validation split, alongside aggregated
rolling cross-validation means, to encourage model selection based on
robustness instead of a single back-test snapshot.

## Reinforcement-Learning KPIs

The RL scaffold reuses the same engineered features but focuses on
reward-driven exploration:

- **Cumulative Reward / Discounted Reward:** Reward functions are
  explicitly configurable (risk-adjusted P&L, directional hits, etc.).
  Sweeps seek positive cumulative reward with stable discounted reward
  trajectories.
- **Sharpe-like Ratio:** Computed on simulated strategy returns to
  gauge the trade-off between aggressiveness and volatility. A Sharpe
  around 0.7–1.0 in simulation is a prerequisite before considering
  cost and slippage modelling.
- **Directional Accuracy & Hit Rate:** Monitor whether the agent’s
  actions align with realised returns and how frequently rewards are
  positive. Values in the mid-50% range are realistic; substantially
  higher readings should be validated via out-of-sample data.
- **Turnover:** Captures trade frequency, allowing us to penalise
  policies that would incur excessive transaction costs.

## Documentation & Governance

- **Experiment tracking:** All runs log to a local MLflow instance under
  `artifacts/mlruns`. Models, metrics, and sweep outcomes are written to
  subdirectories of `artifacts/` for reproducibility.
- **Promotion criteria:** Models or policies must demonstrate consistent
  improvement across both fixed splits and rolling windows, with
  directional accuracy, Sharpe, and turnover remaining within the
  realistic bands outlined above. Blanket targets such as “95% accuracy”
  are intentionally avoided—they are unattainable for noisy financial
  series and often signal data leakage.
- **Future enhancements:** Integrate transaction cost models and
  risk-budgeting constraints once the baseline KPIs exhibit stability
  across multiple market regimes.

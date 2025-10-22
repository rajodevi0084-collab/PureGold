# Risk Management Standards

This document defines the risk management controls for discretionary and systematic strategies routed through KiteConnect. The policies are designed to keep aggregate risk within mandate while ensuring operational resilience during live trading and backtesting.

## Position Sizing Framework
- **Capital-based sizing**: Express target position as a percentage of committed strategy capital. Notional is derived from the most recent close adjusted for expected slippage.
- **Volatility scaling**: Use rolling 20-day annualised volatility to scale target weight so that each position contributes the same marginal risk. Cap position weight when the scaled weight breaches instrument-specific hard limits.
- **Liquidity guardrails**: Never size beyond 10% of the instrument's average traded value for the last 20 sessions. For derivatives, additionally respect the lot-size granularity published by the exchange.
- **Signal gating**: Require signals to clear minimum conviction scores before capital is deployed. Reject trades that would increase cross-asset correlation above the portfolio-level budget.

## Maximum Drawdown Controls
- **Portfolio stop-loss**: Suspend new entries when the strategy experiences a 8% peak-to-trough drawdown over a trailing 30-day window. Resume only after the drawdown recovers to 4%.
- **Instrument-level brakes**: Liquidate positions that fall 5 standard deviations below their trailing 60-day mean return or when realised losses exceed 2.5x the expected volatility contribution.
- **Review cadence**: Trigger an immediate risk committee review for drawdowns exceeding 10% or whenever portfolio VaR breaches 95% of its assigned limit.

## Exposure Limits
- **Gross exposure**: Cap gross exposure at 200% of net asset value, inclusive of futures and options delta-equivalents.
- **Net exposure**: Maintain net exposure within ±50% of NAV. Overnight net exposure must settle within ±30%.
- **Concentration**: Single-name exposure may not exceed 15% of NAV, and sector exposure may not exceed 40% of NAV. Index futures hedges are excluded from sector concentration calculations.
- **Currency and leverage**: Hedge INR/USD exposure when notional USD flows exceed USD 1 million. Monitor margin utilisation daily and maintain a minimum 25% excess margin with KiteConnect brokers.

## Failure-handling Procedures for KiteConnect
- **Order rejections**: Capture rejection codes, reapply idempotent order payloads with exponential back-off (max three attempts), and escalate to the dealing desk if the final attempt fails.
- **Connectivity loss**: Fall back to a read-only session and halt trading when WebSocket heartbeats are missed for 15 seconds. Attempt re-login twice before failing over to the secondary API key.
- **Data discrepancies**: Compare broker confirmation data against local fills. Flag mismatches in the audit log and reconcile before the next trading session.
- **Disaster recovery**: Maintain warm-standby infrastructure in a separate availability zone. Conduct quarterly failover drills to validate KiteConnect credential rotation, session restoration, and reconciliation workflows.

## Governance
- Document all threshold overrides with rationale, owner, and expiry.
- Log every risk-limit evaluation and control trigger to the central audit system.
- Review this policy quarterly or after any major production incident.

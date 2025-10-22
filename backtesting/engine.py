"""Point-in-time backtesting engine with transaction cost and slippage models."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from reference import load_index_constituents, load_symbol_master


@dataclass(slots=True)
class TransactionCostModel:
    """Simple proportional transaction cost model.

    Attributes
    ----------
    bps : float
        Variable cost expressed in basis points of notional traded.
    minimum_fee : float
        Minimum absolute fee per order, applied whenever a trade is executed.
    per_order_fee : float
        Flat brokerage or exchange fee per order.
    """

    bps: float = 0.0
    minimum_fee: float = 0.0
    per_order_fee: float = 0.0

    def estimate(self, notional: float) -> float:
        """Return total cost (in currency) for a trade of the given notional."""

        if notional == 0:
            return 0.0

        variable_cost = abs(notional) * self.bps / 10_000
        total_cost = variable_cost + self.per_order_fee
        if total_cost < self.minimum_fee:
            total_cost = self.minimum_fee
        return total_cost


@dataclass(slots=True)
class SlippageModel:
    """Linear slippage impact model expressed in basis points."""

    bps: float = 0.0

    def apply(self, price: float, direction: float) -> float:
        """Return execution price adjusted for slippage.

        Parameters
        ----------
        price : float
            Reference mid price.
        direction : float
            Sign of the trade (+1 for buy, -1 for sell).
        """

        if direction == 0 or self.bps == 0:
            return price
        return price * (1 + np.sign(direction) * (self.bps / 10_000))


@dataclass(slots=True)
class BacktestConfig:
    """Backtest configuration."""

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    strategy_name: str = "strategy"
    index_name: str | None = None
    report_prefix: str | None = None


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest artefacts."""

    equity_curve: pd.DataFrame
    metrics: dict[str, float | str]
    report_path: Path
    equity_path: Path


class BacktestEngine:
    """Backtesting engine that honours point-in-time universes and trading costs."""

    def __init__(
        self,
        symbol_master_path: Path | str | None = Path("reference/symbol_master.parquet"),
        index_constituents_path: Path | str | None = Path("reference/index_constituents.parquet"),
        transaction_cost_model: TransactionCostModel | None = None,
        slippage_model: SlippageModel | None = None,
        reports_dir: Path | str = Path("reports"),
    ) -> None:
        self.symbol_master = self._load_reference(symbol_master_path, load_symbol_master)
        self.index_constituents = self._load_reference(index_constituents_path, load_index_constituents)
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_reference(
        path: Path | str | None,
        fallback_loader: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        if path is None:
            frame = fallback_loader()
        else:
            reference_path = Path(path)
            if reference_path.exists():
                frame = pd.read_parquet(reference_path)
            else:
                frame = fallback_loader()
        for column in ("start_date", "end_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        return frame

    def select_universe(self, as_of: pd.Timestamp, index_name: str | None = None) -> pd.Index:
        """Return the tradable universe at the provided timestamp."""

        timestamp = pd.Timestamp(as_of)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")

        master = self.symbol_master
        active_mask = (master["start_date"] <= timestamp) & (
            master["end_date"].isna() | (master["end_date"] >= timestamp)
        )
        symbols = master.loc[active_mask, "symbol"]

        if index_name is None:
            return pd.Index(symbols.unique())

        constituents = self.index_constituents
        idx_mask = (constituents["index"] == index_name) & (
            (constituents["start_date"] <= timestamp)
            & (constituents["end_date"].isna() | (constituents["end_date"] >= timestamp))
        )
        allowed = constituents.loc[idx_mask, "symbol"].unique()
        return pd.Index(symbols[symbols.isin(allowed)].unique())

    def run(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Execute the backtest given target weights and price history."""

        signals = signals.copy()
        signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

        signals = signals[(signals["timestamp"] >= start) & (signals["timestamp"] <= end)]
        prices = prices[(prices["timestamp"] >= start) & (prices["timestamp"] <= end)]

        prices = prices.sort_values(["timestamp", "symbol"])
        signals = signals.sort_values(["timestamp", "symbol"])

        timestamps = signals["timestamp"].drop_duplicates().sort_values()
        if timestamps.empty:
            raise ValueError("No signals available within the configured date range.")

        holdings: dict[str, float] = {}
        last_prices: dict[str, float] = {}
        cash = float(config.initial_capital)
        portfolio_values: list[float] = []
        realized_returns: list[float] = []
        curve_index: list[pd.Timestamp] = []

        prev_value = cash

        for ts in timestamps:
            universe = self.select_universe(ts, config.index_name)
            ts_prices = prices[prices["timestamp"] == ts]
            price_map = ts_prices.set_index("symbol")["close"].to_dict()
            last_prices.update(price_map)

            # drop signals for instruments outside the universe
            ts_signals = signals[signals["timestamp"] == ts]
            ts_signals = ts_signals[ts_signals["symbol"].isin(universe)]
            target_weights = ts_signals.set_index("symbol")["target_weight"]

            tradable_symbols = set(universe) | set(holdings)
            pre_trade_value = cash + sum(last_prices[sym] * qty for sym, qty in holdings.items())

            for symbol in sorted(tradable_symbols):
                price = price_map.get(symbol) or last_prices.get(symbol)
                if price is None:
                    continue

                current_shares = holdings.get(symbol, 0.0)
                target_weight = float(target_weights.get(symbol, 0.0))
                target_notional = target_weight * pre_trade_value
                target_shares = target_notional / price if price else 0.0
                trade_shares = target_shares - current_shares

                if abs(trade_shares) < 1e-6:
                    if abs(current_shares) < 1e-6:
                        holdings.pop(symbol, None)
                    else:
                        holdings[symbol] = current_shares
                    continue

                direction = np.sign(trade_shares)
                exec_price = self.slippage_model.apply(price, direction)
                notional = exec_price * trade_shares
                cost = self.transaction_cost_model.estimate(notional)

                cash -= notional
                cash -= cost
                new_position = current_shares + trade_shares
                if abs(new_position) < 1e-6:
                    holdings.pop(symbol, None)
                else:
                    holdings[symbol] = new_position

            portfolio_value = cash + sum(last_prices[sym] * qty for sym, qty in holdings.items())
            curve_index.append(ts)
            portfolio_values.append(portfolio_value)
            period_return = (portfolio_value / prev_value) - 1 if prev_value else 0.0
            realized_returns.append(period_return)
            prev_value = portfolio_value

        equity_curve = pd.DataFrame(
            {
                "timestamp": curve_index,
                "portfolio_value": portfolio_values,
                "period_return": realized_returns,
            }
        ).set_index("timestamp")

        metrics = self._summarise_metrics(equity_curve, config)
        report_path, equity_path = self._persist_reports(equity_curve, metrics, config)

        return BacktestResult(equity_curve=equity_curve, metrics=metrics, report_path=report_path, equity_path=equity_path)

    def _summarise_metrics(self, equity_curve: pd.DataFrame, config: BacktestConfig) -> dict[str, float | str]:
        returns = equity_curve["period_return"].iloc[1:]  # exclude initial period
        total_return = equity_curve["portfolio_value"].iloc[-1] / equity_curve["portfolio_value"].iloc[0] - 1
        std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        volatility = std * np.sqrt(252) if std else 0.0
        mean_return = float(returns.mean()) if not returns.empty else 0.0
        sharpe = (mean_return / std) * np.sqrt(252) if std else 0.0
        max_drawdown = self._max_drawdown(equity_curve["portfolio_value"].to_numpy())

        return {
            "strategy": config.strategy_name,
            "start_date": equity_curve.index.min().isoformat(),
            "end_date": equity_curve.index.max().isoformat(),
            "initial_capital": float(config.initial_capital),
            "final_value": float(equity_curve["portfolio_value"].iloc[-1]),
            "total_return": float(total_return),
            "annualised_volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(max_drawdown),
        }

    @staticmethod
    def _max_drawdown(values: np.ndarray) -> float:
        peak = -np.inf
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value / peak) - 1 if peak > 0 else 0.0
            if drawdown < max_dd:
                max_dd = drawdown
        return abs(max_dd)

    def _persist_reports(
        self,
        equity_curve: pd.DataFrame,
        metrics: dict[str, float | str],
        config: BacktestConfig,
    ) -> tuple[Path, Path]:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        prefix = config.report_prefix or config.strategy_name
        metrics_path = self.reports_dir / f"{prefix}_{timestamp}.json"
        equity_path = self.reports_dir / f"{prefix}_{timestamp}_equity.csv"

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        equity_curve.to_csv(equity_path, index=True)
        return metrics_path, equity_path


__all__ = [
    "TransactionCostModel",
    "SlippageModel",
    "BacktestConfig",
    "BacktestResult",
    "BacktestEngine",
]

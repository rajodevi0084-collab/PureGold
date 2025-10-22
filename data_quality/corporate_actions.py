"""Utilities to maintain corporate action adjustments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, MutableMapping

import polars as pl


@dataclass(slots=True)
class CorporateActionRecord:
    """Adjustment record keyed by symbol and ex-date."""

    symbol: str
    ex_date: date
    split_ratio: float | None = None
    cash_dividend: float | None = None
    close: float | None = None

    @property
    def split_factor(self) -> float:
        return 1.0 / self.split_ratio if self.split_ratio and self.split_ratio != 0 else 1.0

    @property
    def dividend_factor(self) -> float:
        if not self.cash_dividend or not self.close:
            return 1.0
        return 1.0 - (self.cash_dividend / self.close)

    @property
    def adjustment_factor(self) -> float:
        return self.split_factor * self.dividend_factor


class CorporateActionStore:
    """In-memory store for corporate action lookup."""

    def __init__(self) -> None:
        self._records: MutableMapping[tuple[str, date], CorporateActionRecord] = {}

    def load(self, records: Iterable[CorporateActionRecord]) -> None:
        for record in records:
            self._records[(record.symbol, record.ex_date)] = record

    def load_frame(self, frame: pl.DataFrame) -> None:
        required = {"symbol", "ex_date"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"Corporate action frame missing required columns: {sorted(missing)}")

        for row in frame.to_dicts():
            record = CorporateActionRecord(
                symbol=row["symbol"],
                ex_date=row["ex_date"],
                split_ratio=row.get("split_ratio"),
                cash_dividend=row.get("cash_dividend"),
                close=row.get("close"),
            )
            self._records[(record.symbol, record.ex_date)] = record

    def get(self, symbol: str, ex_date: date) -> CorporateActionRecord | None:
        return self._records.get((symbol, ex_date))

    def adjustment_factor(self, symbol: str, ex_date: date) -> float:
        record = self.get(symbol, ex_date)
        return record.adjustment_factor if record else 1.0

    def to_frame(self) -> pl.DataFrame:
        data = [
            {
                "symbol": record.symbol,
                "ex_date": record.ex_date,
                "split_ratio": record.split_ratio,
                "cash_dividend": record.cash_dividend,
                "close": record.close,
                "adjustment_factor": record.adjustment_factor,
            }
            for record in self._records.values()
        ]
        return pl.DataFrame(data)

    def persist(self, path: Path) -> None:
        frame = self.to_frame()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def compute_adjustment_factors(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a frame with computed adjustment factors."""

    required = {"symbol", "ex_date"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Corporate action frame missing required columns: {sorted(missing)}")

    working = frame.clone()
    if "split_ratio" not in working.columns:
        working = working.with_columns(pl.lit(1.0).alias("split_ratio"))
    if "cash_dividend" not in working.columns:
        working = working.with_columns(pl.lit(0.0).alias("cash_dividend"))
    if "close" not in working.columns:
        working = working.with_columns(pl.lit(None).cast(pl.Float64).alias("close"))

    return (
        working.with_columns(
            (
                1.0
                / pl.when(pl.col("split_ratio") == 0)
                .then(1.0)
                .otherwise(pl.col("split_ratio"))
            ).alias("split_factor")
        )
        .with_columns(
            (
                pl.when((pl.col("cash_dividend") == 0) | (pl.col("close").is_null()))
                .then(1.0)
                .otherwise(1.0 - (pl.col("cash_dividend") / pl.col("close")))
            ).alias("dividend_factor")
        )
        .with_columns((pl.col("split_factor") * pl.col("dividend_factor")).alias("adjustment_factor"))
    )


def build_lookup(frame: pl.DataFrame) -> CorporateActionStore:
    store = CorporateActionStore()
    store.load_frame(frame)
    return store

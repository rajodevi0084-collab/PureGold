"""Text-based fixtures for reference datasets.

The previous implementation stored small Parquet files directly in the
repository. Git-based review tooling struggles with binary blobs, so we keep a
textual representation here and expose helpers that can optionally materialise
Parquet snapshots on disk for downstream consumers that expect those paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

_SYMBOL_MASTER_ROWS: list[dict[str, object]] = [
    {
        "symbol": "NSE:INFY",
        "isin": "INE009A01021",
        "exchange": "NSE",
        "name": "Infosys Limited",
        "sector": "Information Technology",
        "start_date": "1993-06-14",
        "end_date": None,
    },
    {
        "symbol": "NSE:TCS",
        "isin": "INE467B01029",
        "exchange": "NSE",
        "name": "Tata Consultancy Services Limited",
        "sector": "Information Technology",
        "start_date": "2004-08-25",
        "end_date": None,
    },
    {
        "symbol": "NSE:HDFCBANK",
        "isin": "INE040A01034",
        "exchange": "NSE",
        "name": "HDFC Bank Limited",
        "sector": "Financials",
        "start_date": "1995-01-23",
        "end_date": None,
    },
]

_INDEX_CONSTITUENT_ROWS: list[dict[str, object]] = [
    {
        "index": "NIFTY50",
        "symbol": "NSE:INFY",
        "weight": 0.085,
        "start_date": "2009-01-01",
        "end_date": None,
    },
    {
        "index": "NIFTY50",
        "symbol": "NSE:TCS",
        "weight": 0.053,
        "start_date": "2009-01-01",
        "end_date": None,
    },
    {
        "index": "NIFTY50",
        "symbol": "NSE:HDFCBANK",
        "weight": 0.112,
        "start_date": "2009-01-01",
        "end_date": None,
    },
    {
        "index": "NIFTYBANK",
        "symbol": "NSE:HDFCBANK",
        "weight": 0.265,
        "start_date": "2013-06-01",
        "end_date": None,
    },
]


def _prepare_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("start_date", "end_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    return frame


def load_symbol_master() -> pd.DataFrame:
    """Return a survivorship-bias-free symbol master snapshot."""

    return _prepare_dates(pd.DataFrame(_SYMBOL_MASTER_ROWS))


def load_index_constituents() -> pd.DataFrame:
    """Return point-in-time index constituent weights."""

    frame = _prepare_dates(pd.DataFrame(_INDEX_CONSTITUENT_ROWS))
    return frame.sort_values(["index", "symbol", "start_date"]).reset_index(drop=True)


def ensure_materialised(base_dir: Path | str = Path("reference")) -> tuple[Path, Path]:
    """Persist Parquet files for consumers that require on-disk artefacts.

    Parameters
    ----------
    base_dir:
        Directory where the Parquet files should live. The default mirrors the
        original repository layout.

    Returns
    -------
    tuple[Path, Path]
        The resolved paths for the symbol master and index constituents files.
    """

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[Path, Callable[[], pd.DataFrame]]] = [
        (base_path / "symbol_master.parquet", load_symbol_master),
        (base_path / "index_constituents.parquet", load_index_constituents),
    ]

    for path, loader in outputs:
        if path.exists():
            continue
        frame = loader()
        try:
            frame.to_parquet(path)
        except (ImportError, ValueError):
            # Parquet support might be optional in lightweight environments.
            # Fall back to a CSV next to the desired artefact for inspection.
            frame.to_csv(path.with_suffix(path.suffix + ".csv"), index=False)

    return tuple(path for path, _ in outputs)

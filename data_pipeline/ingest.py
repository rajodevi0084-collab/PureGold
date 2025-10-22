"""CSV ingestion pipeline using Polars with IST to UTC conversion."""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
from prometheus_client import Counter, Gauge, Histogram, start_http_server

LOGGER = logging.getLogger("data_pipeline.ingest")


ROWS_PROCESSED = Counter(
    "ingest_rows_processed_total",
    "Total number of rows processed by the CSV ingestion job.",
)
INGEST_FAILURES = Counter(
    "ingest_failures_total",
    "Total number of failed ingestion attempts.",
)
INGEST_DURATION = Histogram(
    "ingest_duration_seconds",
    "Duration of each ingestion run in seconds.",
)
LAST_INGEST_TIMESTAMP = Gauge(
    "ingest_last_completed_timestamp",
    "Unix timestamp of the last successful ingestion run.",
)


@dataclass
class IngestConfig:
    """Runtime configuration for the ingestion job."""

    csv_path: Path
    parquet_path: Path
    timestamp_column: str
    zstd_level: int
    metrics_port: int


def _parse_args(argv: Optional[list[str]] = None) -> IngestConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Input CSV file path")
    parser.add_argument("parquet_path", type=Path, help="Destination Parquet file path")
    parser.add_argument(
        "--timestamp-column",
        default=os.getenv("INGEST_TIMESTAMP_COLUMN", "timestamp"),
        help="Name of the timestamp column that should be converted from IST to UTC.",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=int(os.getenv("INGEST_ZSTD_LEVEL", 3)),
        help="Compression level to use when writing Parquet output (ZSTD).",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.getenv("INGEST_METRICS_PORT", 9000)),
        help="Port to expose Prometheus metrics on.",
    )

    args = parser.parse_args(argv)
    return IngestConfig(
        csv_path=args.csv_path,
        parquet_path=args.parquet_path,
        timestamp_column=args.timestamp_column,
        zstd_level=args.zstd_level,
        metrics_port=args.metrics_port,
    )


def _configure_logging() -> None:
    log_level = os.getenv("INGEST_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _convert_ist_to_utc(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        raise KeyError(f"Timestamp column '{column}' not found in CSV data")

    col = pl.col(column)
    try:
        converted = col.dt.convert_time_zone("UTC")
        return frame.with_columns(converted.alias(column))
    except pl.exceptions.PolarsError:
        pass

    try:
        converted = col.dt.replace_time_zone("Asia/Kolkata").dt.convert_time_zone("UTC")
        return frame.with_columns(converted.alias(column))
    except pl.exceptions.PolarsError:
        # Column might be string formatted; parse then convert.
        return frame.with_columns(
            col.strptime(pl.Datetime, strict=False)
            .dt.replace_time_zone("Asia/Kolkata")
            .dt.convert_time_zone("UTC")
            .alias(column)
        )


def _write_parquet(frame: pl.DataFrame, destination: Path, level: int) -> None:
    LOGGER.info("Writing Parquet file to %s with ZSTD level %s", destination, level)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(destination, compression="zstd", compression_level=level)


def ingest(config: IngestConfig) -> None:
    LOGGER.info("Starting ingestion job: csv=%s parquet=%s", config.csv_path, config.parquet_path)
    start_http_server(config.metrics_port)

    start_time = time.time()
    try:
        with INGEST_DURATION.time():
            df = pl.read_csv(config.csv_path, try_parse_dates=True)
            LOGGER.info("Loaded %s rows from CSV", df.height)
            df = _convert_ist_to_utc(df, config.timestamp_column)
            _write_parquet(df, config.parquet_path, config.zstd_level)
            ROWS_PROCESSED.inc(df.height)

        completion_time = time.time()
        LAST_INGEST_TIMESTAMP.set(completion_time)
        LOGGER.info("Completed ingestion in %.2fs", completion_time - start_time)

    except Exception:  # noqa: BLE001 - we need to record metrics on any failure
        INGEST_FAILURES.inc()
        LOGGER.exception("Ingestion failed")
        raise


def _handle_shutdown(signum: int, frame: Optional[object]) -> None:  # noqa: D401, ANN001
    """Handle termination signals for graceful shutdown."""

    LOGGER.info("Received signal %s. Shutting down.", signum)
    sys.exit(0)


def main(argv: Optional[list[str]] = None) -> None:
    _configure_logging()
    config = _parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    ingest(config)


if __name__ == "__main__":
    main()

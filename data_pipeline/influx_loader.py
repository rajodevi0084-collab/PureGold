"""Stream Parquet data into InfluxDB with resilient batching."""
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl
import pyarrow.parquet as pq
from influxdb_client import InfluxDBClient, Point, WriteOptions
from prometheus_client import Counter, Gauge, Histogram, start_http_server

LOGGER = logging.getLogger("data_pipeline.influx_loader")


POINTS_WRITTEN = Counter(
    "influx_loader_points_written_total",
    "Total number of points successfully written to InfluxDB.",
)
WRITE_FAILURES = Counter(
    "influx_loader_failures_total",
    "Total number of failed write attempts to InfluxDB.",
)
WRITE_DURATION = Histogram(
    "influx_loader_batch_duration_seconds",
    "Time spent writing each batch of points to InfluxDB.",
)
LAST_SUCCESS_EPOCH = Gauge(
    "influx_loader_last_success_timestamp",
    "Unix timestamp of the last successful write batch.",
)
CURRENT_BATCH_SIZE = Gauge(
    "influx_loader_current_batch_size",
    "Number of points in the batch currently being processed.",
)


@dataclass
class LoaderConfig:
    parquet_path: Path
    bucket: str
    org: str
    token: str
    url: str
    measurement: str
    timestamp_column: str
    tag_columns: tuple[str, ...]
    batch_size: int
    metrics_port: int


def _parse_args(argv: Optional[list[str]] = None) -> LoaderConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_path", type=Path, help="Source Parquet file to stream from.")
    parser.add_argument("--bucket", required=True, help="InfluxDB bucket to write into.")
    parser.add_argument("--org", required=True, help="InfluxDB organisation identifier.")
    parser.add_argument("--token", default=os.getenv("INFLUX_TOKEN"), help="InfluxDB access token.")
    parser.add_argument(
        "--url", default=os.getenv("INFLUX_URL", "http://localhost:8086"), help="InfluxDB base URL."
    )
    parser.add_argument(
        "--measurement",
        default=os.getenv("INFLUX_MEASUREMENT", "metrics"),
        help="Measurement name to use for all points.",
    )
    parser.add_argument(
        "--timestamp-column",
        default=os.getenv("INFLUX_TIMESTAMP_COLUMN", "timestamp"),
        help="Name of the timestamp column.",
    )
    parser.add_argument(
        "--tag-columns",
        default=os.getenv("INFLUX_TAG_COLUMNS", ""),
        help="Comma separated list of tag columns.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("INFLUX_BATCH_SIZE", 5000)),
        help="Number of points to accumulate before writing to InfluxDB.",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.getenv("INFLUX_METRICS_PORT", 9001)),
        help="Port to expose Prometheus metrics on.",
    )

    args = parser.parse_args(argv)
    tag_columns = tuple(filter(None, (column.strip() for column in args.tag_columns.split(","))))

    missing = [name for name, value in (("token", args.token),) if value is None]
    if missing:
        parser.error(f"Missing required credentials: {', '.join(missing)}")

    return LoaderConfig(
        parquet_path=args.parquet_path,
        bucket=args.bucket,
        org=args.org,
        token=args.token,
        url=args.url,
        measurement=args.measurement,
        timestamp_column=args.timestamp_column,
        tag_columns=tag_columns,
        batch_size=args.batch_size,
        metrics_port=args.metrics_port,
    )


def _configure_logging() -> None:
    log_level = os.getenv("INFLUX_LOADER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _iter_record_batches(parquet_path: Path, batch_size: int) -> Iterable[pl.DataFrame]:
    file = pq.ParquetFile(parquet_path)
    for batch in file.iter_batches(batch_size=batch_size):
        yield pl.from_arrow(batch)


def _rows_to_points(
    rows: Iterable[dict[str, object]],
    measurement: str,
    timestamp_column: str,
    tag_columns: tuple[str, ...],
) -> list[Point]:
    points: list[Point] = []
    for row in rows:
        timestamp = row.get(timestamp_column)
        if timestamp is None:
            raise ValueError(f"Row missing timestamp column '{timestamp_column}'")

        point = Point(measurement)
        if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is not None:
            point.time(timestamp)
        else:
            # Assume naive timestamp is UTC.
            point.time(timestamp, write_precision="ns")

        for tag in tag_columns:
            value = row.get(tag)
            if value is not None:
                point.tag(tag, str(value))

        for field, value in row.items():
            if field in tag_columns or field == timestamp_column:
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                point.field(field, value)
            elif isinstance(value, (int, float)):
                if isinstance(value, float) and math.isnan(value):
                    continue
                point.field(field, value)
            else:
                point.field(field, str(value))

        points.append(point)

    return points


def stream_to_influx(config: LoaderConfig) -> None:
    LOGGER.info(
        "Streaming Parquet data to InfluxDB: file=%s bucket=%s measurement=%s",
        config.parquet_path,
        config.bucket,
        config.measurement,
    )
    start_http_server(config.metrics_port)

    options = WriteOptions(
        batch_size=config.batch_size,
        flush_interval=1000,
        jitter_interval=500,
        retry_interval=5000,
        max_retries=5,
        max_retry_delay=30000,
        exponential_base=2,
    )

    with InfluxDBClient(url=config.url, token=config.token, org=config.org) as client:
        write_api = client.write_api(write_options=options)

        for frame in _iter_record_batches(config.parquet_path, config.batch_size):
            rows = frame.to_dicts()
            CURRENT_BATCH_SIZE.set(len(rows))
            points = _rows_to_points(rows, config.measurement, config.timestamp_column, config.tag_columns)
            if not points:
                CURRENT_BATCH_SIZE.set(0)
                continue

            try:
                with WRITE_DURATION.time():
                    write_api.write(bucket=config.bucket, record=points)
                POINTS_WRITTEN.inc(len(points))
                LAST_SUCCESS_EPOCH.set(time.time())
            except Exception:  # noqa: BLE001
                WRITE_FAILURES.inc()
                LOGGER.exception("Failed to write batch to InfluxDB")
                raise
            finally:
                CURRENT_BATCH_SIZE.set(0)


def _handle_shutdown(signum: int, frame: Optional[object]) -> None:  # noqa: D401, ANN001
    """Handle termination signals for graceful shutdown."""

    LOGGER.info("Received signal %s. Shutting down.", signum)
    sys.exit(0)


def main(argv: Optional[list[str]] = None) -> None:
    _configure_logging()
    config = _parse_args(argv)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    stream_to_influx(config)


if __name__ == "__main__":
    main()

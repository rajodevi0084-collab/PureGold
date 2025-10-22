"""CSV ingestion pipeline using Polars with data quality enforcement."""
from __future__ import annotations

import argparse
import hashlib
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

from data_quality import schemas, validators

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
    dataset: str
    expected_frequency: str
    duplicate_keys: tuple[str, ...]
    duplicate_alert_threshold: int
    gap_alert_threshold: int
    outlier_columns: tuple[str, ...]
    outlier_alert_threshold: int
    mad_threshold: float
    metadata_dsn: str | None
    audit_log_path: Path
    alert_webhook: str | None
    quarantine_dir: Path


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
    parser.add_argument(
        "--dataset",
        default=os.getenv("INGEST_DATASET", "market_data"),
        help="Named dataset schema to enforce.",
    )
    parser.add_argument(
        "--expected-frequency",
        default=os.getenv("INGEST_EXPECTED_FREQUENCY", "1m"),
        help="Expected spacing used for gap detection (e.g. 1m, 5m, 1h).",
    )
    parser.add_argument(
        "--duplicate-keys",
        default=os.getenv("INGEST_DUPLICATE_KEYS", "timestamp,symbol"),
        help="Comma separated key columns for duplicate collapse.",
    )
    parser.add_argument(
        "--duplicate-alert-threshold",
        type=int,
        default=int(os.getenv("INGEST_DUPLICATE_ALERT_THRESHOLD", 0)),
        help="Trigger alerts when duplicate count exceeds this threshold.",
    )
    parser.add_argument(
        "--gap-alert-threshold",
        type=int,
        default=int(os.getenv("INGEST_GAP_ALERT_THRESHOLD", 0)),
        help="Trigger alerts when gap count exceeds this threshold.",
    )
    parser.add_argument(
        "--outlier-columns",
        default=os.getenv("INGEST_OUTLIER_COLUMNS", "close"),
        help="Comma separated numeric columns to evaluate for MAD outliers.",
    )
    parser.add_argument(
        "--outlier-alert-threshold",
        type=int,
        default=int(os.getenv("INGEST_OUTLIER_ALERT_THRESHOLD", 0)),
        help="Trigger alerts when quarantined outliers exceed this threshold.",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=float(os.getenv("INGEST_MAD_THRESHOLD", 3.5)),
        help="Modified z-score threshold for MAD outlier detection.",
    )
    parser.add_argument(
        "--audit-log-path",
        type=Path,
        default=Path(os.getenv("AUDIT_LOG_PATH", "logs/audit.log")),
        help="Location for structured audit logs.",
    )
    parser.add_argument(
        "--alert-webhook",
        default=os.getenv("VALIDATION_ALERT_WEBHOOK"),
        help="Optional webhook endpoint for validation alerts.",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=Path(os.getenv("OUTLIER_QUARANTINE_DIR", "logs/quarantine")),
        help="Directory for persisting quarantined outliers.",
    )

    args = parser.parse_args(argv)
    duplicate_keys = tuple(filter(None, (key.strip() for key in args.duplicate_keys.split(","))))
    outlier_columns = tuple(filter(None, (col.strip() for col in args.outlier_columns.split(","))))

    return IngestConfig(
        csv_path=args.csv_path,
        parquet_path=args.parquet_path,
        timestamp_column=args.timestamp_column,
        zstd_level=args.zstd_level,
        metrics_port=args.metrics_port,
        dataset=args.dataset,
        expected_frequency=args.expected_frequency,
        duplicate_keys=duplicate_keys,
        duplicate_alert_threshold=args.duplicate_alert_threshold,
        gap_alert_threshold=args.gap_alert_threshold,
        outlier_columns=outlier_columns,
        outlier_alert_threshold=args.outlier_alert_threshold,
        mad_threshold=args.mad_threshold,
        metadata_dsn=os.getenv("INGEST_METADATA_DSN"),
        audit_log_path=args.audit_log_path,
        alert_webhook=args.alert_webhook,
        quarantine_dir=args.quarantine_dir,
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


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _import_psycopg():  # type: ignore[return-type]
    try:
        import psycopg  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "psycopg is required for ingestion metadata tracking but is not installed"
        ) from exc
    return psycopg


def _ingestion_hash_exists(dsn: str, source_hash: str) -> bool:
    psycopg = _import_psycopg()
    with psycopg.connect(dsn) as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM ingestion_runs WHERE source_hash = %s",
                (source_hash,),
            )
            return cursor.fetchone() is not None


def _record_ingestion_run(
    dsn: str,
    *,
    config: IngestConfig,
    source_hash: str,
    row_count: int,
) -> int:
    psycopg = _import_psycopg()
    with psycopg.connect(dsn) as conn:  # type: ignore[attr-defined]
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_runs (
                    source_path,
                    destination_path,
                    source_hash,
                    row_count,
                    status
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_hash) DO UPDATE
                SET destination_path = EXCLUDED.destination_path,
                    row_count = EXCLUDED.row_count,
                    status = EXCLUDED.status,
                    completed_at = NOW()
                RETURNING id
                """,
                (
                    str(config.csv_path),
                    str(config.parquet_path),
                    source_hash,
                    row_count,
                    "completed",
                ),
            )
            run_id = cursor.fetchone()[0]
        conn.commit()
    return run_id


def ingest(config: IngestConfig) -> None:
    LOGGER.info("Starting ingestion job: csv=%s parquet=%s", config.csv_path, config.parquet_path)
    start_http_server(config.metrics_port)

    audit_logger = validators.AuditLogger(
        "data_pipeline.ingest",
        log_path=config.audit_log_path,
        alert_webhook=config.alert_webhook,
    )

    source_hash = _compute_sha256(config.csv_path)
    audit_logger.log(
        "ingestion_started",
        csv_path=str(config.csv_path),
        parquet_path=str(config.parquet_path),
        source_hash=source_hash,
    )

    if config.metadata_dsn:
        LOGGER.info("Checking idempotency metadata store for hash %s", source_hash)
        if _ingestion_hash_exists(config.metadata_dsn, source_hash):
            LOGGER.warning("Source file already ingested (hash match). Skipping run.")
            audit_logger.log(
                "ingestion_skipped_duplicate",
                level=logging.WARNING,
                source_hash=source_hash,
                csv_path=str(config.csv_path),
                parquet_path=str(config.parquet_path),
            )
            audit_logger.alert(
                "ingestion_skipped_duplicate",
                severity="warning",
                source_hash=source_hash,
                csv_path=str(config.csv_path),
            )
            return

    start_time = time.time()
    try:
        with INGEST_DURATION.time():
            df = pl.read_csv(config.csv_path, try_parse_dates=True)
            LOGGER.info("Loaded %s rows from CSV", df.height)

            df = _convert_ist_to_utc(df, config.timestamp_column)

            schema = schemas.get_schema(config.dataset)
            diff = schemas.diff_frame(schema, df)
            if not diff.ok:
                audit_logger.log(
                    "schema_validation_failed",
                    level=logging.ERROR,
                    dataset=config.dataset,
                    diff=diff.to_dict(),
                )
                audit_logger.alert(
                    "schema_validation_failed",
                    severity="error",
                    dataset=config.dataset,
                    diff=diff.to_dict(),
                )
                raise ValueError(f"Schema validation failed for dataset {config.dataset}")

            gap_result = validators.detect_time_gaps(
                df,
                timestamp_column=config.timestamp_column,
                expected_frequency=config.expected_frequency,
                audit_logger=audit_logger,
                alert_threshold=config.gap_alert_threshold,
            )
            if gap_result.count:
                LOGGER.warning("Detected %s gaps greater than %s", gap_result.count, gap_result.expected_frequency)

            df = validators.collapse_duplicates(
                df,
                subset=config.duplicate_keys,
                order_by=(config.timestamp_column,) if config.timestamp_column else None,
                audit_logger=audit_logger,
                alert_threshold=config.duplicate_alert_threshold,
            )

            df, quarantined = validators.mad_outlier_quarantine(
                df,
                columns=config.outlier_columns,
                threshold=config.mad_threshold,
                audit_logger=audit_logger,
                alert_threshold=config.outlier_alert_threshold,
            )

            if quarantined.height:
                config.quarantine_dir.mkdir(parents=True, exist_ok=True)
                quarantine_path = (
                    config.quarantine_dir
                    / f"{config.csv_path.stem}_outliers_{int(time.time())}.parquet"
                )
                quarantined.write_parquet(quarantine_path)
                audit_logger.log(
                    "mad_outliers_persisted",
                    path=str(quarantine_path),
                    total_quarantined=quarantined.height,
                )

            df = schemas.enforce_schema(df, schema)

            _write_parquet(df, config.parquet_path, config.zstd_level)
            ROWS_PROCESSED.inc(df.height)

        completion_time = time.time()
        LAST_INGEST_TIMESTAMP.set(completion_time)
        duration = completion_time - start_time
        LOGGER.info("Completed ingestion in %.2fs", duration)
        audit_logger.log(
            "ingestion_completed",
            duration_seconds=duration,
            row_count=df.height,
            source_hash=source_hash,
        )

        if config.metadata_dsn:
            run_id = _record_ingestion_run(
                config.metadata_dsn,
                config=config,
                source_hash=source_hash,
                row_count=df.height,
            )
            audit_logger.log(
                "ingestion_metadata_recorded",
                run_id=run_id,
                row_count=df.height,
                source_hash=source_hash,
            )

    except Exception:  # noqa: BLE001 - record metrics on failure
        INGEST_FAILURES.inc()
        audit_logger.alert(
            "ingestion_failed",
            severity="error",
            csv_path=str(config.csv_path),
            parquet_path=str(config.parquet_path),
        )
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

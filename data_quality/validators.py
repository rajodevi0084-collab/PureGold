"""Validation utilities for data quality enforcement."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

import polars as pl

LOGGER = logging.getLogger("data_quality.validators")


def _default_json_serializer(value: object) -> str:
    if isinstance(value, (datetime, timedelta)):
        return value.isoformat() if isinstance(value, datetime) else value.total_seconds()
    return str(value)


class AuditLogger:
    """Structured JSON audit logging with optional webhook alerts."""

    def __init__(
        self,
        application: str,
        *,
        log_path: Path | None = None,
        alert_webhook: str | None = None,
    ) -> None:
        self.application = application
        self.log_path = log_path or Path("logs/audit.log")
        self.alert_webhook = alert_webhook or os.getenv("VALIDATION_ALERT_WEBHOOK")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger_name = f"audit.{application}.{id(self)}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def log(self, event: str, *, level: int = logging.INFO, **details: object) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "application": self.application,
            "event": event,
            "details": details,
            "level": logging.getLevelName(level),
        }
        message = json.dumps(payload, default=_default_json_serializer, sort_keys=True)
        self._logger.log(level, message)

    def alert(self, event: str, *, severity: str = "warning", **details: object) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "application": self.application,
            "event": event,
            "severity": severity,
            "details": details,
        }
        message = json.dumps(payload, default=_default_json_serializer, sort_keys=True)
        LOGGER.warning("Validation alert triggered: %s", message)

        if not self.alert_webhook:
            return

        try:
            request = Request(
                self.alert_webhook,
                data=message.encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5):
                pass
        except URLError:
            LOGGER.exception("Failed to dispatch alert to webhook")


def _to_timedelta(expected_frequency: str | timedelta) -> timedelta:
    if isinstance(expected_frequency, timedelta):
        return expected_frequency

    value = expected_frequency.strip().lower()
    if value.endswith("ms"):
        amount = float(value[:-2])
        return timedelta(milliseconds=amount)
    unit = value[-1]
    amount = float(value[:-1])
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "d":
        return timedelta(days=amount)
    raise ValueError(f"Unsupported frequency specifier: {expected_frequency}")


@dataclass(slots=True)
class GapDetectionResult:
    """Summary of detected time gaps."""

    gaps: pl.DataFrame
    expected_frequency: timedelta

    @property
    def count(self) -> int:
        return self.gaps.height


def detect_time_gaps(
    frame: pl.DataFrame,
    *,
    timestamp_column: str,
    expected_frequency: str | timedelta,
    audit_logger: AuditLogger | None = None,
    alert_threshold: int = 0,
) -> GapDetectionResult:
    """Return gaps that exceed the expected frequency."""

    if timestamp_column not in frame.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not present in frame")

    frequency = _to_timedelta(expected_frequency)
    timestamps = frame.sort(timestamp_column).get_column(timestamp_column).to_list()

    records: list[dict[str, object]] = []
    for current, nxt in zip(timestamps, timestamps[1:]):
        if current is None or nxt is None:
            continue
        delta = nxt - current
        if delta > frequency:
            missing_intervals = max(int(delta / frequency) - 1, 0)
            records.append(
                {
                    "start_timestamp": current,
                    "end_timestamp": nxt,
                    "gap": delta,
                    "missing_intervals": missing_intervals,
                }
            )

    if records:
        gap_frame = pl.DataFrame(records)
    else:
        gap_frame = pl.DataFrame(
            {
                "start_timestamp": pl.Series(name="start_timestamp", values=[], dtype=pl.Datetime),
                "end_timestamp": pl.Series(name="end_timestamp", values=[], dtype=pl.Datetime),
                "gap": pl.Series(
                    name="gap", values=[], dtype=pl.Duration(time_unit="ns")
                ),
                "missing_intervals": pl.Series(
                    name="missing_intervals", values=[], dtype=pl.Int64
                ),
            }
        )

    if audit_logger and records:
        audit_logger.log(
            "time_gaps_detected",
            count=len(records),
            expected_frequency=frequency,
            sample=records[:10],
        )
        if len(records) > alert_threshold:
            audit_logger.alert(
                "time_gaps_detected",
                severity="error",
                count=len(records),
                expected_frequency=str(frequency),
            )

    return GapDetectionResult(gap_frame, frequency)


def collapse_duplicates(
    frame: pl.DataFrame,
    *,
    subset: Sequence[str],
    order_by: Sequence[str] | None = None,
    audit_logger: AuditLogger | None = None,
    alert_threshold: int = 0,
) -> pl.DataFrame:
    """Collapse duplicate rows keeping the most recent observation."""

    if not subset:
        return frame

    subset_list = list(subset)
    order_list = list(order_by) if order_by else None

    working = frame.sort(order_list) if order_list else frame

    key_counts = working.group_by(subset_list).agg(pl.len().alias("_count"))
    duplicate_keys = key_counts.filter(pl.col("_count") > 1).drop("_count")
    if duplicate_keys.is_empty():
        duplicates = working.head(0)
    else:
        duplicates = working.join(duplicate_keys, on=subset_list, how="inner")

    if audit_logger and duplicates.height > 0:
        audit_logger.log(
            "duplicates_collapsed",
            keys=subset_list,
            collapsed_count=duplicates.height,
            sample=duplicates.head(10).to_dicts(),
        )
        if duplicates.height > alert_threshold:
            audit_logger.alert(
                "duplicates_collapsed",
                severity="error",
                keys=subset_list,
                collapsed_count=duplicates.height,
            )

    deduped = working.unique(
        subset=subset_list,
        keep="last" if order_list else "first",
        maintain_order=bool(order_list),
    )
    return deduped.sort(order_list) if order_list else deduped


def mad_outlier_quarantine(
    frame: pl.DataFrame,
    *,
    columns: Iterable[str],
    threshold: float = 3.5,
    audit_logger: AuditLogger | None = None,
    alert_threshold: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a frame into clean data and MAD-based outliers."""

    if not columns:
        return frame, frame.head(0)

    mask = pl.Series(name="is_outlier", values=[False] * frame.height)
    metrics: dict[str, dict[str, float]] = {}

    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].cast(pl.Float64, strict=False)
        valid = series.drop_nulls()
        if valid.is_empty():
            continue

        median = valid.median()
        abs_dev = (series - median).abs()
        mad = abs_dev.drop_nulls().median()
        if mad == 0 or mad is None:
            continue
        scores = (abs_dev / mad) * 0.67448975
        outlier_mask = scores > threshold
        outlier_mask = outlier_mask.fill_null(False)
        mask = mask | outlier_mask
        metrics[column] = {
            "median": float(median),
            "mad": float(mad),
            "outliers": int(outlier_mask.sum()),
        }

    cleaned = frame.filter(~mask)
    quarantined = frame.filter(mask)

    if audit_logger and quarantined.height > 0:
        audit_logger.log(
            "mad_outliers_quarantined",
            total_quarantined=quarantined.height,
            columns=metrics,
            threshold=threshold,
        )
        if quarantined.height > alert_threshold:
            audit_logger.alert(
                "mad_outliers_quarantined",
                severity="error",
                total_quarantined=quarantined.height,
                columns=metrics,
            )

    return cleaned, quarantined

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from data_quality import corporate_actions, schemas, validators


class SchemaTests(unittest.TestCase):
    def test_diff_frame_detects_missing_and_mismatched(self) -> None:
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "symbol": ["PG"],
                "open": ["1.0"],
                "close": [1.2],
                "volume": [100],
            }
        )

        schema = schemas.get_schema("market_data")
        diff = schemas.diff_frame(schema, frame)

        self.assertIn("high", diff.missing_columns)
        self.assertIn("low", diff.missing_columns)
        self.assertIn("trade_count", diff.missing_columns)
        self.assertIn("open", diff.dtype_mismatches)
        self.assertFalse(diff.ok)

    def test_enforce_schema_orders_and_casts(self) -> None:
        frame = pl.DataFrame(
            {
                "symbol": ["PG"],
                "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "open": ["1.0"],
                "high": ["1.1"],
                "low": ["0.9"],
                "close": ["1.2"],
                "volume": ["100"],
                "trade_count": ["10"],
            }
        )

        schema = schemas.get_schema("market_data")
        enforced = schemas.enforce_schema(frame, schema)

        self.assertEqual(tuple(enforced.columns), schema.column_names())
        self.assertEqual(enforced.schema["open"], pl.Float64)
        self.assertEqual(enforced.schema["volume"], pl.Int64)


class ValidatorTests(unittest.TestCase):
    def test_detect_time_gaps_returns_gap_frame(self) -> None:
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 9, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 9, 5, tzinfo=timezone.utc),
                ]
            }
        )

        result = validators.detect_time_gaps(
            frame, timestamp_column="timestamp", expected_frequency="1m"
        )

        self.assertEqual(result.count, 1)
        self.assertAlmostEqual(
            result.gaps["missing_intervals"].item(),
            3,
        )

    def test_collapse_duplicates_prefers_latest_by_timestamp(self) -> None:
        frame = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                ],
                "symbol": ["PG", "PG"],
                "close": [1.0, 1.5],
            }
        )

        deduped = validators.collapse_duplicates(
            frame,
            subset=("timestamp", "symbol"),
            order_by=("timestamp",),
        )

        self.assertEqual(deduped.height, 1)
        self.assertEqual(deduped["close"].item(), 1.5)

    def test_mad_outlier_quarantine_flags_extreme_values(self) -> None:
        frame = pl.DataFrame(
            {
                "close": [100.0, 101.0, 99.5, 250.0],
            }
        )

        clean, quarantined = validators.mad_outlier_quarantine(
            frame,
            columns=["close"],
            threshold=3.5,
        )

        self.assertEqual(quarantined.height, 1)
        self.assertEqual(quarantined["close"].item(), 250.0)
        self.assertEqual(clean.height, 3)

    def test_audit_logger_writes_structured_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.log"
            logger = validators.AuditLogger("test_app", log_path=log_path)

            logger.log("test_event", details={"value": 1})

            contents = log_path.read_text().strip()
            payload = json.loads(contents)

            self.assertEqual(payload["application"], "test_app")
            self.assertEqual(payload["event"], "test_event")
            self.assertEqual(payload["details"], {"details": {"value": 1}})


class CorporateActionsTests(unittest.TestCase):
    def test_compute_adjustment_factors(self) -> None:
        frame = pl.DataFrame(
            {
                "symbol": ["PG"],
                "ex_date": [datetime(2024, 1, 10, tzinfo=timezone.utc).date()],
                "split_ratio": [2.0],
                "cash_dividend": [1.0],
                "close": [10.0],
            }
        )

        adjusted = corporate_actions.compute_adjustment_factors(frame)
        self.assertAlmostEqual(adjusted["split_factor"].item(), 0.5)
        self.assertAlmostEqual(adjusted["dividend_factor"].item(), 0.9)
        self.assertAlmostEqual(adjusted["adjustment_factor"].item(), 0.45)

    def test_store_lookup_and_persist(self) -> None:
        record = corporate_actions.CorporateActionRecord(
            symbol="PG",
            ex_date=datetime(2024, 1, 10, tzinfo=timezone.utc).date(),
            split_ratio=2.0,
            cash_dividend=1.0,
            close=10.0,
        )

        store = corporate_actions.CorporateActionStore()
        store.load([record])

        self.assertAlmostEqual(store.adjustment_factor("PG", record.ex_date), 0.45)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corp.parquet"
            store.persist(path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()

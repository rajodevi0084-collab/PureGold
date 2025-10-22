# Performance Benchmarks

This document captures the repeatable methodology we use to benchmark the
CSV ingestion and InfluxDB loading pipelines. Follow these steps whenever
validating changes to ingestion logic, infrastructure, or environment tuning.

## Test Data

* Use a synthetic dataset sized to at least 10 million rows with realistic
  column distributions (timestamps at 1s cadence, high-cardinality tags,
  and mixed numeric/string fields).
* Store the canonical CSV under `./benchmark_data/telemetry.csv` and the
  corresponding Parquet output under `./benchmark_data/telemetry.parquet`.
* Regenerate the dataset whenever schema changes are introduced so cold and
  warm runs stay comparable.

## Environment Preparation

1. Provision a dedicated benchmark host (Apple M2 Pro or equivalent) with
   Python 3.11, Docker, and the dependencies from `pyproject.toml`.
2. Start the local InfluxDB stack using `docker compose up -d` from
   `infrastructure/influxdb/`. Allow the database to finish startup before
   running benchmarks.
3. Ensure no other heavy workloads are running on the host, and pin the
   benchmark process to performance CPU cores when possible.

## Cold Run Procedure

1. Clear the Parquet output directory and drop the InfluxDB bucket so the
   run starts from a cold cache.
2. Restart the ingestion metrics server to reset counters.
3. Execute the ingestion pipeline:
   ```bash
   python -m data_pipeline.ingest ./benchmark_data/telemetry.csv \
       ./benchmark_data/telemetry.parquet \
       --timestamp-column timestamp --zstd-level 7
   ```
4. Record `ingest_duration_seconds` and the computed rows/second rate.
5. Stream the resulting Parquet into InfluxDB:
   ```bash
   python -m data_pipeline.influx_loader ./benchmark_data/telemetry.parquet \
       --bucket telemetry --org puregold --token $INFLUX_TOKEN \
       --measurement signals --timestamp-column timestamp \
       --tag-columns device_id,region --batch-size 10000
   ```
6. Capture `influx_loader_batch_duration_seconds` histogram samples and
   calculate the sustained points/second throughput.

## Warm Run Procedure

1. Repeat the ingestion and loading steps immediately after the cold run
   without restarting services.
2. Because the dataset and caches are now hot, the focus is on stability—
   verify the throughput stays within 5% of the cold run results.
3. Export Prometheus metrics from both services for archival.

## Acceptance Thresholds

To sign off on a change, both cold and warm runs must meet the following:

* **CSV ingestion throughput:** ≥ 200,000 rows/second measured over the full run.
* **InfluxDB loading throughput:** ≥ 100,000 points/second sustained during the
  warm run.
* **Error budget:** zero failed batches (`*_failures_total` counters must stay 0).

If any metric falls below target, investigate compression levels, batch sizes,
network/disk contention, or regressions introduced by recent code changes.

## Reporting

Document each benchmark in the engineering journal with:

* Git commit SHA of the code under test.
* Hardware and OS details of the benchmark host.
* Metric snapshots and Prometheus exports.
* Observations or anomalies discovered during the run.

These records allow trend tracking and highlight when deeper profiling is
required.

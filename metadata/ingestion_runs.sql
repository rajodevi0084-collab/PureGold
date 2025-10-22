-- Migration: ingestion_runs metadata table for idempotency tracking
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id SERIAL PRIMARY KEY,
    source_path TEXT NOT NULL,
    destination_path TEXT NOT NULL,
    source_hash TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ingestion_runs_source_path_idx
    ON ingestion_runs (source_path);

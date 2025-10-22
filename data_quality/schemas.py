"""Schema registry and validation helpers for incoming datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Definition for a single column in a golden schema."""

    name: str
    dtype: pl.datatypes.DataType
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class TableSchema:
    """Golden schema definition for a named dataset."""

    name: str
    columns: tuple[ColumnSchema, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def as_polars_schema(self) -> dict[str, pl.datatypes.DataType]:
        return {column.name: column.dtype for column in self.columns}


@dataclass(slots=True)
class SchemaDiff:
    """Differences between a golden schema and an observed frame."""

    missing_columns: dict[str, ColumnSchema]
    unexpected_columns: dict[str, pl.datatypes.DataType]
    dtype_mismatches: dict[str, tuple[pl.datatypes.DataType, pl.datatypes.DataType]]
    non_nullable_violations: set[str]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_columns
            and not self.unexpected_columns
            and not self.dtype_mismatches
            and not self.non_nullable_violations
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "missing_columns": {name: str(column.dtype) for name, column in self.missing_columns.items()},
            "unexpected_columns": {name: str(dtype) for name, dtype in self.unexpected_columns.items()},
            "dtype_mismatches": {
                name: {
                    "expected": str(expected),
                    "observed": str(observed),
                }
                for name, (expected, observed) in self.dtype_mismatches.items()
            },
            "non_nullable_violations": sorted(self.non_nullable_violations),
        }


GOLDEN_SCHEMAS: dict[str, TableSchema] = {
    "market_data": TableSchema(
        name="market_data",
        columns=(
            ColumnSchema("timestamp", pl.Datetime(time_zone="UTC"), nullable=False),
            ColumnSchema("symbol", pl.Utf8, nullable=False),
            ColumnSchema("open", pl.Float64),
            ColumnSchema("high", pl.Float64),
            ColumnSchema("low", pl.Float64),
            ColumnSchema("close", pl.Float64),
            ColumnSchema("volume", pl.Int64),
            ColumnSchema("trade_count", pl.Int64),
        ),
    ),
    "corporate_actions": TableSchema(
        name="corporate_actions",
        columns=(
            ColumnSchema("symbol", pl.Utf8, nullable=False),
            ColumnSchema("ex_date", pl.Date, nullable=False),
            ColumnSchema("split_ratio", pl.Float64),
            ColumnSchema("cash_dividend", pl.Float64),
            ColumnSchema("close", pl.Float64),
        ),
    ),
}


def register_schema(schema: TableSchema) -> None:
    """Register or override a schema at runtime."""

    GOLDEN_SCHEMAS[schema.name] = schema


def get_schema(name: str) -> TableSchema:
    """Fetch a golden schema by name."""

    try:
        return GOLDEN_SCHEMAS[name]
    except KeyError as exc:
        raise KeyError(f"No schema registered under name '{name}'") from exc


def diff_frame(schema: TableSchema, frame: pl.DataFrame) -> SchemaDiff:
    """Compare a Polars frame with the expected schema."""

    expected = {column.name: column for column in schema.columns}
    observed = frame.schema

    missing: dict[str, ColumnSchema] = {}
    unexpected: dict[str, pl.datatypes.DataType] = {}
    mismatches: dict[str, tuple[pl.datatypes.DataType, pl.datatypes.DataType]] = {}
    non_nullable_violations: set[str] = set()

    for name, column_schema in expected.items():
        if name not in observed:
            missing[name] = column_schema
            continue

        observed_dtype = observed[name]
        if observed_dtype != column_schema.dtype:
            mismatches[name] = (column_schema.dtype, observed_dtype)

        if not column_schema.nullable:
            series = frame.get_column(name)
            if series.null_count() > 0:
                non_nullable_violations.add(name)

    for name, dtype in observed.items():
        if name not in expected:
            unexpected[name] = dtype

    return SchemaDiff(missing, unexpected, mismatches, non_nullable_violations)


def validate_frame(schema: TableSchema, frame: pl.DataFrame) -> SchemaDiff:
    """Validate a frame against the schema, raising on errors."""

    diff = diff_frame(schema, frame)
    if not diff.ok:
        raise ValueError(f"Schema validation failed for '{schema.name}': {diff.to_dict()}")
    return diff


def diff_file(path: Path, schema: TableSchema, *, read_csv_kwargs: Mapping[str, object] | None = None) -> SchemaDiff:
    """Load a file and diff its schema against the golden definition."""

    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    extension = path.suffix.lower()
    if extension == ".csv":
        frame = pl.read_csv(path, **read_csv_kwargs)
    elif extension == ".parquet":
        frame = pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file extension for schema diff: {extension}")

    return diff_frame(schema, frame)


def enforce_schema(frame: pl.DataFrame, schema: TableSchema) -> pl.DataFrame:
    """Return a frame with columns ordered and cast to the golden schema."""

    diff = diff_frame(schema, frame)
    blocking_issues = {}
    if diff.missing_columns:
        blocking_issues["missing_columns"] = {
            name: str(column.dtype) for name, column in diff.missing_columns.items()
        }
    if diff.unexpected_columns:
        blocking_issues["unexpected_columns"] = {
            name: str(dtype) for name, dtype in diff.unexpected_columns.items()
        }
    if diff.non_nullable_violations:
        blocking_issues["non_nullable_violations"] = sorted(diff.non_nullable_violations)

    if blocking_issues:
        raise ValueError(
            f"Cannot enforce schema '{schema.name}' due to structural issues: {blocking_issues}"
        )

    expressions = []
    for column in schema.columns:
        expr = pl.col(column.name)
        if frame.schema[column.name] != column.dtype:
            expr = expr.cast(column.dtype)
        expressions.append(expr.alias(column.name))

    return frame.select(expressions)


def list_registered_schemas() -> Iterable[str]:
    """Yield the names of registered schemas."""

    return GOLDEN_SCHEMAS.keys()

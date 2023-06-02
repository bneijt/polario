from typing import Literal, Optional
from urllib.parse import urlsplit

import fsspec
import polars as pl


class DeltaDataset:
    """Dataset based on deltatable storage"""

    def __init__(self, url: str, partition_columns: list[str] = []):
        self.url = url.rstrip("/")
        # Load fsspec filesystem from url scheme
        location = urlsplit(url)
        self.fs = fsspec.filesystem(location.scheme)
        self.partition_columns = partition_columns

    def append(self, df: pl.DataFrame) -> None:
        self._write_delta(df, mode="append")

    def write(self, df: pl.DataFrame) -> None:
        self._write_delta(df, mode="overwrite")

    def _write_delta(
        self, df: pl.DataFrame, mode: Literal["append", "overwrite"]
    ) -> None:
        if not set(df.columns).issuperset(set(self.partition_columns)) or len(
            df.columns
        ) <= len(self.partition_columns):
            raise ValueError(
                f"Dataframe should have more columns, require at least {self.partition_columns}, got {df.columns}"
            )
        df.write_delta(
            self.url,
            mode=mode,
            delta_write_options={"partition_by": self.partition_columns},
        )

    def read_partition(self, partition_column_values: dict[str, str]) -> pl.DataFrame:
        if set(partition_column_values.keys()) != set(self.partition_columns):
            raise ValueError(
                f"Partition column value keys {partition_column_values} do not match partition columns {self.partition_columns}"
            )

        return pl.read_delta(
            self.url,
            pyarrow_options={
                "partitions": [
                    (pcol, "=", pval) for pcol, pval in partition_column_values.items()
                ]
            },
        )

    def scan(self) -> Optional[pl.LazyFrame]:
        try:
            return pl.scan_delta(self.url)
        except Exception as e:
            if ".TableNotFoundError" in str(type(e)):
                return None
            raise e

"""The hive dataset implementation"""
from collections import OrderedDict
from functools import reduce
from typing import Iterable, Type
from urllib.parse import urlsplit
from uuid import uuid4

import fsspec
import polars as pl
from fsspec.spec import AbstractFileSystem


class ParquetFragment:
    """Pointer to a parquet fragment"""

    def __init__(self, url: str):
        self.url = url

    @classmethod
    def first_fragment(
        cls: Type["ParquetFragment"], base_url: str
    ) -> "ParquetFragment":
        """Return a fragment name for what should be the first fragment in the partition"""
        idx = 0
        return cls(f"{base_url}/{idx:06}_{uuid4().hex}.parquet")

    def next_fragment(self) -> "ParquetFragment":
        """Return a fragment name for what should be the next fragment in the partition"""
        idx = int(self.url.split("/")[-1].split("_")[0]) + 1
        return ParquetFragment(
            f"{self.url.split('/')[:-1]}/{idx:06}_{uuid4().hex}.parquet"
        )

    def read(self) -> pl.DataFrame:
        """Read the fragment"""
        return pl.read_parquet(self.url)

    def scan(self) -> pl.LazyFrame:
        """Read the fragment"""
        return pl.scan_parquet(self.url)

    def write(self, df: pl.DataFrame) -> None:
        """Write the fragment"""
        df.to_parquet(self.url)


class HivePartition:
    """Pointer to a partition in a HiveDataset"""

    def __init__(
        self,
        fs: AbstractFileSystem,
        base_url: str,
        partition_column_values: OrderedDict[str, str],
    ) -> None:
        self.fs = fs
        self.base_url = base_url
        self.partition_column_values = partition_column_values

    @classmethod
    def from_relative_path(
        cls: Type["HivePartition"], relative_path: str
    ) -> "HivePartition":
        """Create a partition from a relative path"""
        return cls(OrderedDict(relative_path.split("/").map(lambda x: x.split("="))))

    def relative_path(self) -> str:
        """Create a relative path from a partition"""
        return "/".join([f"{k}={v}" for k, v in self.partition_column_values.items()])

    def fragment_urls(self) -> Iterable[str]:
        return self.fs.expand_path(
            "/".join([self.base_url, self.relative_path(), "*.parquet"])
        )

    def fragments(self) -> Iterable[ParquetFragment]:
        """Discover fragments"""
        return map(ParquetFragment, self.fragment_urls())

    def read(self) -> pl.DataFrame:
        """Concat the fragments in this partition into a single dataframe"""
        return pl.concat([f.read() for f in self.fragments()]).with_columns(
            *map(
                lambda pcol, pval: pl.lit(pval).alias(pcol),
                self.partition_column_values.items(),
            )
        )

    def scan(self) -> pl.LazyFrame:
        """Concat the fragments in this partition into a single dataframe"""
        return pl.concat([f.scan() for f in self.fragments()]).with_columns(
            *map(
                lambda pcol, pval: pl.lit(pval).alias(pcol),
                self.partition_column_values.items(),
            )
        )

    def write(self, df: pl.DataFrame) -> None:
        """Write the dataframe to this partition"""

        if self.fs.exists(self.base_url):
            self.fs.delete(self.base_url, recursive=True)

        target_fragment = ParquetFragment.first_fragment(self.base_url)
        for fragment_df in df.iter_slices(1e6):
            target_fragment.write(fragment_df)
            target_fragment = target_fragment.next_fragment()

    def append(self, df: pl.DataFrame) -> None:
        """Write the dataframe to this partition"""

        new_fragment = ParquetFragment.first_fragment(self.base_url)
        try:
            *_, last_fragment = self.fragments()
            new_fragment = last_fragment.next_fragment()
        except ValueError:
            pass
        new_fragment.write(df)


class HiveDataset:
    """Handle to multiple partitions"""

    def __init__(self, base_url: str, partition_columns: list[str]) -> None:
        self.base_url = base_url
        self.location = urlsplit(base_url)
        self.fs = fsspec.filesystem(self.location.scheme)
        self.partition_columns = partition_columns

    def partitions(self) -> Iterable[HivePartition]:
        """Iterate over HivePartitions"""
        glob_pattern = HivePartition(
            OrderedDict({k: "*" for k in self.partition_columns})
        ).to_relative_path()
        partitions = self.fs.expand_path(glob_pattern)
        return map(HivePartition.from_relative_path, sorted(partitions))

    def read_partitions(self) -> Iterable[pl.DataFrame]:
        """Iterate over partitions"""
        return map(HivePartition.read, self.partitions())

    def scan_partitions(self) -> Iterable[pl.LazyFrame]:
        """Iterate over partitions"""
        return map(HivePartition.scan, self.partitions())

    def scan(self) -> pl.LazyFrame:
        return pl.concat(self.scan_partitions())

    def read(self) -> pl.DataFrame:
        return pl.concat(self.read_partitions())

    def write(self, df: pl.DataFrame) -> None:
        # Split dataframe into partition values
        partition_values = df.select(self.partition_columns).unique().to_dicts()
        # Write each partition
        for partition_value in partition_values:
            partition = HivePartition(
                OrderedDict(
                    [partition_value.get(pcol) for pcol in self.partition_columns]
                )
            )
            partition.write(
                df.filter(
                    reduce(
                        lambda a, b: a & b,
                        [pl.col(k) == v for k, v in partition_value.items()],
                    )
                )
            )

    def append(self, df: pl.DataFrame) -> None:
        partition_values = df.select(self.partition_columns).unique().to_dicts()
        # Write each partition
        for partition_value in partition_values:
            partition = HivePartition(
                OrderedDict(
                    [partition_value.get(pcol) for pcol in self.partition_columns]
                )
            )
            partition.append(
                df.filter(
                    reduce(
                        lambda a, b: a & b,
                        [pl.col(k) == v for k, v in partition_value.items()],
                    )
                )
            )

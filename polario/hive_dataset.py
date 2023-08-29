"""The hive dataset implementation"""
from collections import OrderedDict
from functools import reduce
from itertools import chain
from typing import Iterable, Optional, Tuple, Type
from urllib.parse import urlsplit
from uuid import uuid4

import fsspec
import polars as pl
from fsspec.spec import AbstractFileSystem

DEFAULT_ROWS_PER_FRAGMENT = int(1e6)

DEFAULT_PARQUET_WRITE_OPTIONS = {
    "use_pyarrow": True,
    "compression": "snappy",
}


def to_relative_location_from(
    possible_prefix: str, base_location: str, location: str
) -> str:
    """Take a location and make it relative to the base location, stripping possible prefix from both"""
    relative_location = location
    if location.startswith(possible_prefix):
        relative_location = relative_location[len(possible_prefix) :]

    # If base location is not absolute, it might be somewhere in location
    if not base_location.startswith("/"):
        if base_location in relative_location:
            relative_location = relative_location[
                relative_location.find(base_location) :
            ]

    relative_location = relative_location.lstrip("/")
    scheme_less_url = base_location[len(possible_prefix) :].lstrip("/")
    if relative_location.startswith(scheme_less_url):
        relative_location = relative_location[len(scheme_less_url) + 1 :]
    return relative_location


class ParquetFragment:
    """Pointer to a parquet fragment"""

    def __init__(self, url: str, parquet_write_options: dict):
        self.url = url
        self.parquet_write_options = parquet_write_options

    @classmethod
    def first_fragment(
        cls: Type["ParquetFragment"],
        partition_base_url: str,
        parquet_write_options: dict,
    ) -> "ParquetFragment":
        """Return a fragment name for what should be the first fragment in the partition"""
        idx = 0
        return cls(
            f"{partition_base_url}/{idx:06}_{uuid4().hex}.parquet",
            parquet_write_options,
        )

    def next_fragment(self) -> "ParquetFragment":
        """Return a fragment name for what should be the next fragment in the partition"""
        idx = int(self.url.split("/")[-1].split("_")[0]) + 1
        return ParquetFragment(
            f"{'/'.join(self.url.split('/')[:-1])}/{idx:06}_{uuid4().hex}.parquet",
            self.parquet_write_options,
        )

    def read(self) -> pl.DataFrame:
        """Read the fragment"""
        return pl.read_parquet(self.url)

    def scan(self) -> pl.LazyFrame:
        """Read the fragment"""
        return pl.scan_parquet(self.url)

    def write(self, df: pl.DataFrame) -> None:
        """Write the fragment"""
        df.write_parquet(self.url, **self.parquet_write_options)


class HivePartition:
    """Pointer to a partition in a HiveDataset"""

    def __init__(
        self,
        fs: AbstractFileSystem,
        dataset_url: str,
        partition_column_values: OrderedDict[str, str],
        maximum_rows_per_fragment: int,
        parquet_write_options: dict,
    ) -> None:
        self.fs = fs
        self.partition_column_values = partition_column_values
        self.maximum_rows_per_fragment = maximum_rows_per_fragment
        self.url = (dataset_url + "/" + self.to_relative_path()).rstrip("/")
        location = urlsplit(dataset_url)
        self.scheme_prefix = location.scheme + "://" if location.scheme else ""
        self._parquet_write_options = parquet_write_options

    @classmethod
    def from_relative_path(
        cls: Type["HivePartition"],
        fs: AbstractFileSystem,
        dataset_url: str,
        relative_path: str,
        maximum_rows_per_fragment: int,
        parquet_write_options: dict,
    ) -> "HivePartition":
        """Create a partition from a relative path"""
        relative_path_elements = relative_path.split("/")
        if any(map(lambda x: "=" not in x, relative_path_elements)):
            raise ValueError(
                f"One or more parition path elements is missing an equal sign while parsing '{relative_path}' from '{dataset_url}'"
            )

        return cls(
            fs=fs,
            dataset_url=dataset_url,
            partition_column_values=OrderedDict(
                map(lambda x: x.split("=", 1), relative_path_elements)
            ),
            maximum_rows_per_fragment=maximum_rows_per_fragment,
            parquet_write_options=parquet_write_options,
        )

    def to_relative_path(self) -> str:
        """Create a relative path from partition column values"""
        return "/".join([f"{k}={v}" for k, v in self.partition_column_values.items()])

    def fragment_urls(self) -> Iterable[str]:
        try:
            return map(
                lambda p: self.url
                + "/"
                + to_relative_location_from(self.scheme_prefix, self.url, p),
                self.fs.expand_path("/".join([self.url, "*.parquet"])),
            )
        except FileNotFoundError:
            return []

    def fragments(self) -> Iterable[ParquetFragment]:
        """Discover fragments"""
        return map(
            lambda fragment_url: ParquetFragment(
                fragment_url,
                self._parquet_write_options,
            ),
            self.fragment_urls(),
        )

    def read(self) -> Optional[pl.DataFrame]:
        """Concat the fragments in this partition into a single dataframe"""
        fragments = [f.read() for f in self.fragments()]
        if len(fragments) > 1:
            # Merge schemas from different fragments into a superset schema
            superset_schema: dict[str, pl.PolarsDataType] = reduce(
                lambda a, b: a | dict(b.schema),
                fragments[1:],
                dict(fragments[0].schema),
            )

            def add_missing_columns(df: pl.DataFrame) -> pl.DataFrame:
                missing_columns = superset_schema.keys() - set(df.columns)
                if missing_columns:
                    return df.with_columns(
                        [
                            pl.lit(None, superset_schema[col]).alias(col)
                            for col in missing_columns
                        ]
                    )
                else:
                    return df

            complete_fragements = [
                add_missing_columns(f).select(superset_schema.keys()) for f in fragments
            ]
        else:
            complete_fragements = fragments

        if complete_fragements:
            return pl.concat(complete_fragements).with_columns(
                map(
                    lambda part: pl.lit(part[1]).alias(part[0]),
                    self.partition_column_values.items(),
                )
            )
        return None

    def scan(self) -> Optional[pl.LazyFrame]:
        """Concat the fragments in this partition into a single dataframe"""
        fragments = [f.scan() for f in self.fragments()]
        if fragments:
            return pl.concat(fragments).with_columns(
                map(
                    lambda part: pl.lit(part[1]).alias(part[0]),
                    self.partition_column_values.items(),
                )
            )
        return None

    def _write_to_fragments(
        self, df: pl.DataFrame, target_fragment: ParquetFragment
    ) -> None:
        output_columns = list(
            sorted(set(df.columns) - self.partition_column_values.keys())
        )
        for fragment_df in df.select(output_columns).iter_slices(
            self.maximum_rows_per_fragment
        ):
            target_fragment.write(fragment_df)
            target_fragment = target_fragment.next_fragment()

    def delete(self) -> None:
        """Delete the partition"""
        if self.fs.exists(self.url):
            self.fs.delete(self.url, recursive=True)

    def write(self, df: pl.DataFrame) -> None:
        """Write the dataframe to this partition"""
        self.delete()
        self.fs.mkdir(self.url)
        target_fragment = ParquetFragment.first_fragment(
            self.url, self._parquet_write_options
        )
        self._write_to_fragments(df, target_fragment)

    def append(self, df: pl.DataFrame) -> None:
        """Write the dataframe to this partition"""
        if not self.fs.exists(self.url):
            self.fs.mkdir(self.url)

        new_fragment = ParquetFragment.first_fragment(
            self.url, self._parquet_write_options
        )
        try:
            *_, last_fragment = self.fragments()
            new_fragment = last_fragment.next_fragment()
        except ValueError:
            pass
        self._write_to_fragments(df, new_fragment)


class HiveDataset:
    """Handle to multiple partitions"""

    def __init__(
        self,
        url: str,
        partition_columns: list[str] = [],
        max_rows_per_fragment: int = DEFAULT_ROWS_PER_FRAGMENT,
        parquet_write_options: dict = DEFAULT_PARQUET_WRITE_OPTIONS,
    ) -> None:
        self.url = url.rstrip("/")
        # Load fsspec filesystem from url scheme
        location = urlsplit(url)
        self.fs = fsspec.filesystem(location.scheme)
        self.scheme_prefix = location.scheme + "://" if location.scheme else ""
        self.partition_columns = partition_columns
        self._max_rows_per_fragment = max_rows_per_fragment
        self._parquet_write_options = parquet_write_options

    def partitions(self) -> Iterable[HivePartition]:
        """Iterate over HivePartitions"""
        if self.partition_columns:
            glob_pattern = HivePartition(
                fs=self.fs,
                dataset_url=self.url,
                partition_column_values=OrderedDict(
                    {k: "*" for k in self.partition_columns}
                ),
                maximum_rows_per_fragment=self._max_rows_per_fragment,
                parquet_write_options=self._parquet_write_options,
            ).to_relative_path()
            try:
                partitions = self.fs.expand_path(self.url + "/" + glob_pattern)

                return map(
                    lambda path: HivePartition.from_relative_path(
                        fs=self.fs,
                        dataset_url=self.url,
                        relative_path=to_relative_location_from(
                            self.scheme_prefix, self.url, path
                        ),
                        maximum_rows_per_fragment=self._max_rows_per_fragment,
                        parquet_write_options=self._parquet_write_options,
                    ),
                    sorted(partitions),
                )
            except FileNotFoundError:
                return []
        else:
            return [
                HivePartition(
                    fs=self.fs,
                    dataset_url=self.url,
                    partition_column_values=OrderedDict(),
                    maximum_rows_per_fragment=self._max_rows_per_fragment,
                    parquet_write_options=self._parquet_write_options,
                )
            ]

    def read_partitions(self) -> Iterable[pl.DataFrame]:
        """Iterate over partitions"""
        for partition in self.partitions():
            df = partition.read()
            if df is not None:
                yield df

    def read_partition(
        self, partition_column_values: dict[str, str]
    ) -> Optional[pl.DataFrame]:
        """Read the given partition from the dataset"""
        if set(partition_column_values.keys()) != set(self.partition_columns):
            raise ValueError(
                f"Partition column value keys {partition_column_values} do not match partition columns {self.partition_columns}"
            )
        return HivePartition(
            fs=self.fs,
            dataset_url=self.url,
            partition_column_values=OrderedDict(partition_column_values),
            maximum_rows_per_fragment=self._max_rows_per_fragment,
            parquet_write_options=self._parquet_write_options,
        ).read()

    def delete_partition(self, partition_column_values: dict[str, str]) -> None:
        """Read the given partition from the dataset"""
        if set(partition_column_values.keys()) != set(self.partition_columns):
            raise ValueError(
                f"Partition column value keys {partition_column_values} do not match partition columns {self.partition_columns}"
            )
        return HivePartition(
            fs=self.fs,
            dataset_url=self.url,
            partition_column_values=OrderedDict(partition_column_values),
            maximum_rows_per_fragment=self._max_rows_per_fragment,
            parquet_write_options=self._parquet_write_options,
        ).delete()

    def scan_partitions(self) -> Iterable[pl.LazyFrame]:
        """Iterate over partitions"""
        for partition in self.partitions():
            df = partition.scan()
            if df is not None:
                yield df

    def scan(self) -> Optional[pl.LazyFrame]:
        iterable = iter(self.scan_partitions())
        first_partition = next(iterable, None)
        if first_partition is not None:
            return pl.concat(chain([first_partition], iterable))
        return None

    def _check_partition_columns(self, df: pl.DataFrame) -> None:
        """Check if the given dataframe fits in this dataset"""
        if not set(df.columns).issuperset(set(self.partition_columns)) or len(
            df.columns
        ) <= len(self.partition_columns):
            raise ValueError(
                f"Dataframe should have more columns, require at least {self.partition_columns}, got {df.columns}"
            )
        for pcol in self.partition_columns:
            if df[pcol].dtype != pl.Utf8:
                raise ValueError(
                    f"Partition column {pcol} is not a string, but {df[pcol].dtype}"
                )

    def _partition_split(
        self, df: pl.DataFrame
    ) -> Iterable[Tuple[pl.DataFrame, HivePartition]]:
        """Split dataframe into partitions and partition dataframes"""
        self._check_partition_columns(df)
        if self.partition_columns == []:
            yield df, HivePartition(
                fs=self.fs,
                dataset_url=self.url,
                partition_column_values=OrderedDict(),
                maximum_rows_per_fragment=self._max_rows_per_fragment,
                parquet_write_options=self._parquet_write_options,
            )
        else:
            partition_values = df.select(self.partition_columns).unique().to_dicts()
            for partition_value in partition_values:
                yield df.filter(
                    reduce(
                        lambda a, b: a & b,
                        [pl.col(k) == v for k, v in partition_value.items()],
                    )
                ), HivePartition(
                    fs=self.fs,
                    dataset_url=self.url,
                    partition_column_values=OrderedDict(
                        [
                            (pcol, partition_value[pcol])
                            for pcol in self.partition_columns
                        ]
                    ),
                    maximum_rows_per_fragment=self._max_rows_per_fragment,
                    parquet_write_options=self._parquet_write_options,
                )

    def write(self, df: pl.DataFrame) -> None:
        self._check_partition_columns(df)
        for partition_df, partition in self._partition_split(df):
            partition.write(partition_df)

    def append(self, df: pl.DataFrame) -> None:
        self._check_partition_columns(df)
        for partition_df, partition in self._partition_split(df):
            partition.append(partition_df)

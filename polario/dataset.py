"""The main dataset package"""
from enum import Enum
from functools import reduce
from typing import Iterable, Literal, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import fsspec
import polars as pl
import pyarrow as pa
import pyarrow.dataset


class ExistingDataBehavior(Enum):
    """Controls how the dataset will handle data that already exists in
    the destination. See `pyarrow.dataset.write_dataset` for more information
    """

    ERROR = "error"
    """raise an error if any data exists in the destination"""
    OVERWRITE_OR_IGNORE = "overwrite_or_ignore"
    """ignore any existing data and will overwrite files with the same name as an output file"""
    DELETE_MATCHING = "delete_matching"
    """The first time each partition directory is encountered the entire directory will be deleted"""


class HiveDataset:
    """Dataset handle for a Hive partitioned set of Parquet files"""

    def __init__(
        self,
        base_url: str,
        partition_columns: Optional[list[str]] = None,
        max_rows_per_file=1048576,
    ):
        """Create a dataset handle at the given configuration

        Args:
            base_url (str): The url the dataset is located, the scheme of this url will be tested against fsspec to check if it is accessible.
            partition_columns (Optional[list[str]], optional): List of columns names used for partitioning. If this list if empty or None, no partitioning will be performed. Defaults to None.
            max_rows_per_file (int, optional): The maximum number of rows per parquet file on disk. Defaults to 1048576.
        """
        self.location = urlsplit(base_url)
        fsspec.filesystem(self.location.scheme)
        self.partition_columns = partition_columns or []
        self.max_rows_per_file = max_rows_per_file

    def pyarrow_dataset(self) -> pyarrow.dataset.Dataset:
        """Get hive flavoured pyarrow dataset from dataset location"""
        return pyarrow.dataset.dataset(
            self.location.geturl(),
            partitioning=pyarrow.dataset.partitioning(
                schema=pa.schema(
                    [
                        (partition_column, pa.string())
                        for partition_column in self.partition_columns
                    ]
                ),
                flavor="hive",
            ),
        )

    def read(self) -> pl.DataFrame:
        """Read the whole dataset into a dataframe"""
        return self.scan().collect()

    def to_compatible_arrow_table(self, data_frame: pl.DataFrame) -> pa.Table:
        """Transforms the given data frame into a `pyarrow.Table` that has
        compatible columns that allow the partition columns to be written to a filesystem.

        Args:
            data_frame (pl.DataFrame): The data frame to load

        Raises:
            ValueError: If the data frame contains no non-partition columns

        Returns:
            pa.Table: A table with the partition columns cast to simple strings
        """
        table = data_frame.to_arrow()

        if self.partition_columns:
            if len(self.partition_columns) >= len(data_frame.columns):
                raise ValueError(
                    f"Can not have all columns as partition columns, got '{self.partition_columns}' as partition columns and a dataframe with '{data_frame.columns}'."
                )
            # Cast all partition columns to string values
            table = table.cast(
                pyarrow.schema(
                    [
                        f.with_type(pyarrow.string())
                        if f.name in self.partition_columns
                        else f
                        for f in table.schema
                    ]
                )
            )
        return table

    def write(
        self,
        data_frame: pl.DataFrame,
        existing_data_behavior: ExistingDataBehavior = ExistingDataBehavior.DELETE_MATCHING,
    ) -> None:
        """Write the given dataframe to the dataset

        The default behavior expects you to write dataframes with full partitions.

        Args:
            dataframe (pl.DataFrame): DataFrame to write to this dataset
            existing_data_behavior (ExistingDataBehavior, optional): What to do when a partition from the dataframe already exists. Defaults to ExistingDataBehavior.DELETE_MATCHING.

        Raises:
            ValueError: When the partition columns are not strings, see `HiveDataset.assert_partition_columns_are_string`
        """
        self.assert_partition_columns_are_string(data_frame)
        table = self.to_compatible_arrow_table(data_frame)
        pyarrow.dataset.write_dataset(
            table,
            self.location.geturl(),
            format="parquet",
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            existing_data_behavior=existing_data_behavior.value,
            max_rows_per_group=self.max_rows_per_file,
            max_rows_per_file=self.max_rows_per_file,
        )

    def scan(self) -> pl.LazyFrame:
        return pl.scan_ds(self.pyarrow_dataset())

    def read_partitions(self) -> Iterable[pl.DataFrame]:
        """Scan all partitions of the dataset into separate dataframes

        Returns:
            Iterable[pl.DataFrame]: A partition
        """
        ds = self.pyarrow_dataset()
        partition_expressions = {
            str(fragment.partition_expression): fragment.partition_expression
            for fragment in ds.get_fragments()
        }
        for partition_expression in partition_expressions.values():
            yield pl.DataFrame(ds.filter(partition_expression).to_table())

    def read_partition(self, partition_values: dict[str, str]) -> pl.DataFrame:
        """Read a single partition without doing a scan of the whole dataset. This is efficient, but also quite hacky.

        Args:
            partition_values (dict[str, str]): A dictionary of values of each partition column

        Returns:
            pl.DataFrame: The read dataframe
        Raises:
            ValueError: When a value for a partition column is missing
            FileNotFoundError: When the requested partition does not exist
        """
        partition_location = self.partition_location(partition_values=partition_values)
        fs = fsspec.filesystem(self.location.scheme)
        parquet_files = [
            fragment_location
            for fragment_location in fs.ls(partition_location, detail=False)
            if fragment_location.endswith(".parquet")
        ]
        scheme_prefix = f"{self.location.scheme}://" if self.location.scheme else ""
        return pl.concat(
            [
                pl.read_parquet(f"{scheme_prefix}{parquet_file}")
                for parquet_file in parquet_files
            ]
        ).with_columns(
            [
                pl.lit(parcol_value).alias(parcol_name)
                for parcol_name, parcol_value in partition_values.items()
            ]
        )

    def partition_location(self, partition_values: dict[str, str]) -> str:
        """Generate a hive partitioning url based on the dataset location and the given partition values.
        This method requires all partition columns to be mentioned and have a value.

        Args:
            partition_values (dict[str, str]): Partition column values

        Returns:
            str: Location of the hive partition
        Raises:
            ValueError: When a value for a partition column is missing
        """
        if not set(self.partition_columns) == partition_values.keys():
            raise ValueError(
                f"All partition columns need to be part of the partition selection, got {partition_values.keys()}, require {self.partition_columns}"
            )
        path_elements = [
            f"{pc}={partition_values.get(pc)}" for pc in self.partition_columns
        ]
        ploc = self.location._replace(
            path="/".join([self.location.path] + path_elements)
        )
        return urlunsplit(ploc)

    def assert_partition_columns_are_string(self, df: pl.DataFrame) -> None:
        """Raise a value error if any of the partition columns are not `polars.Utf8`

        This is important, because the partition column values are only stored on the filesystem
        as directory names, this means they will always be cast to string and read back as string again.

        Args:
            df (pl.DataFrame): Dataframe to check
        Raises:
            ValueError: When the partition columns are not of type `polars.Utf8`.
        """
        wrong_columns = [
            (col_name, col_type)
            for col_name, col_type in df.schema.items()
            if col_name in self.partition_columns and col_type != pl.Utf8
        ]
        if wrong_columns:
            raise ValueError(
                f"Partition columns must be of type string for dataset. Wrong columns are {wrong_columns}"
            )

    def update(
        self,
        other_df: pl.DataFrame,
        on: Sequence[str],
        how: Literal["left", "inner"] = "left",
    ) -> None:
        """Upsert the given dataframe into the dataset.

        This will partition the dataframe and for each partition, load the partition, merge
        the dataframe and write back to the dataset.

        Args:
            upsert_df (pl.Dataframe): Dataframe to upsert into the dataset
            on (pl.Dataframe): Dataframe to upsert into the dataset
        Raises:
            ValueError: When the partition columns are not of type `polars.Utf8`
            ValueError: When you try to update a partition that is not in the dataset
        """
        self.assert_partition_columns_are_string(other_df)
        partitions = other_df.select(self.partition_columns).unique()
        for partition_values in partitions.to_dicts():
            # Filter other_df on partition values for the partition we are about to update
            partition_matching_expressions = [
                pl.col(pcol) == pval for pcol, pval in partition_values.items()
            ]
            other_partition_df = other_df.filter(
                reduce(
                    lambda a, b: a & b,
                    partition_matching_expressions[1:],
                    partition_matching_expressions[0],
                )
            )
            try:
                updated_partition_df = self.read_partition(
                    partition_values=partition_values
                ).update(other_partition_df, on=on, how=how)
                self.write(updated_partition_df)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Unable to update partition that is not in the dataset. other_df contains {partition_values}",
                    e,
                )

    def append(self, data_frame: pl.DataFrame) -> None:
        """Append the given data frame to the dataset

        Args:
            data_frame (pl.DataFrame): Data frame to add to the dataset
        """
        self.assert_partition_columns_are_string(data_frame)
        table = self.to_compatible_arrow_table(data_frame)
        pyarrow.dataset.write_dataset(
            table,
            self.location.geturl(),
            format="parquet",
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            existing_data_behavior=ExistingDataBehavior.OVERWRITE_OR_IGNORE.value,
            max_rows_per_group=self.max_rows_per_file,
            max_rows_per_file=self.max_rows_per_file,
            basename_template=uuid4().hex + "_{i}.parquet",
        )

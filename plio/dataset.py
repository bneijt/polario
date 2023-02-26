from typing import Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

import fsspec
import polars as pl
import pyarrow as pa
import pyarrow.dataset


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

    def write(
        self,
        dataframe: pl.DataFrame,
        existing_data_behavior: str = "delete_matching",
    ) -> None:
        table = dataframe.to_arrow()

        if self.partition_columns:
            if len(self.partition_columns) >= len(dataframe.columns):
                raise ValueError(
                    f"Can not have all columns as partition columns, got '{self.partition_columns}' as partition columns and a dataframe with '{dataframe.columns}'."
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
        pyarrow.dataset.write_dataset(
            table,
            self.location.geturl(),
            format="parquet",
            partitioning=self.partition_columns,
            partitioning_flavor="hive",
            existing_data_behavior=existing_data_behavior,
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
        """
        partition_location = self.partition_location(partition_values=partition_values)
        fs = fsspec.filesystem(self.location.scheme)
        parquet_files = [
            fragment_location
            for fragment_location in fs.ls(partition_location, detail=False)
            if fragment_location.endswith(".parquet")
        ]
        return pl.concat(
            [pl.read_parquet(parquet_file) for parquet_file in parquet_files]
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

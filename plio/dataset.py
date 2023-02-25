from typing import Iterable
from urllib.parse import urlsplit
import fsspec
import polars as pl
import pyarrow.dataset
from typing import Optional
import pyarrow as pa


class Dataset:
    """Dataset handle for a hive parquet dataset"""

    def __init__(self, base_url: str, partition_columns: Optional[list[str]] = None, max_rows_per_file = 1048576):
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
        MAX_ROWS_PER_GROUP = 1024 * 1024
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


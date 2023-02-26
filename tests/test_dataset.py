import os
import tempfile
from typing import Iterable

import polars as pl
import pytest

from plio.dataset import HiveDataset


@pytest.fixture
def example_df_1() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {"p1": 1, "p2": "a", "v": 1},
            {"p1": 1, "p2": "b", "v": 1},
            {"p1": 2, "p2": "a", "v": 1},
            {"p1": 2, "p2": "a", "v": 1},
        ]
    )


@pytest.fixture
def example_ds_1(example_df_1: pl.DataFrame) -> Iterable[HiveDataset]:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = HiveDataset(tempdir, partition_columns=["p1", "p2"], max_rows_per_file=1)
        ds.write(example_df_1)
        yield ds


def test_dataset_should_raise_for_unsupported_protocol() -> None:
    with pytest.raises(ValueError):
        HiveDataset("example://some/url")


def test_write_partitioned_data(example_df_1: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        HiveDataset(tempdir).write(example_df_1)
        assert "part-0.parquet" in os.listdir(
            tempdir
        ), "Should write out one top-level part-0.parquet"

    with tempfile.TemporaryDirectory() as tempdir:
        HiveDataset(tempdir, partition_columns=["p1"]).write(example_df_1)
        assert "p1=1" in os.listdir(tempdir), "Should partition with p1"

    with tempfile.TemporaryDirectory() as tempdir:
        HiveDataset(tempdir, partition_columns=["p1", "p2"]).write(example_df_1)
        assert "p1=1" in os.listdir(tempdir), "Should partition with a first"
        assert "p2=a" in os.listdir(f"{tempdir}/p1=1"), "Should partition with b second"


def test_read_partitions(example_ds_1: HiveDataset, example_df_1: pl.DataFrame) -> None:
    partitions = list(example_ds_1.read_partitions())
    assert (
        len(partitions) == 3
    ), "Three different combinations of a and b columns values"
    assert [p.shape for p in partitions] == [
        (1, 3),
        (1, 3),
        (2, 3),
    ], "Should have read the right shapes"
    assert (
        pl.concat(partitions).to_dicts()
        == example_df_1.with_columns([pl.col("p1").cast(pl.Utf8)]).to_dicts()
    ), "Partition columns are read back as Utf8"


def test_partion_columns_should_be_string(example_ds_1: HiveDataset) -> None:
    df = example_ds_1.read()
    assert df.schema == {"v": pl.Int64, "p1": pl.Utf8, "p2": pl.Utf8}
    assert df.shape == (4, 3)


def test_read_partion_should_read_single_partition(example_ds_1: HiveDataset) -> None:
    with pytest.raises(ValueError):
        # Missing p2
        example_ds_1.read_partition({"p1": "1"})
    with pytest.raises(FileNotFoundError):
        # Missing p2
        example_ds_1.read_partition({"p1": "not_there", "p2": "not_there"})

    partition_df = example_ds_1.read_partition({"p1": "1", "p2": "a"})
    assert partition_df.shape == (
        1,
        3,
    ), "This partition should contain only a single row and all columns"
    partition_df = example_ds_1.read_partition({"p1": "2", "p2": "a"})
    assert partition_df.shape == (
        2,
        3,
    ), "This partition should contain two rows and all columns"

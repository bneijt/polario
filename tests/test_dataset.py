from plio.dataset import Dataset
import pytest
import polars as pl
import os
import tempfile
from typing import Iterable


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
def example_ds_1(example_df_1: pl.DataFrame) -> Iterable[Dataset]:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1", "p2"])
        ds.write(example_df_1)
        yield ds


def test_dataset_should_raise_for_unsupported_protocol() -> None:
    with pytest.raises(ValueError):
        Dataset("example://some/url")


def test_write_partitioned_data(example_df_1: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        Dataset(tempdir).write(example_df_1)
        assert "part-0.parquet" in os.listdir(
            tempdir
        ), "Should write out one top-level part-0.parquet"

    with tempfile.TemporaryDirectory() as tempdir:
        Dataset(tempdir, partition_columns=["p1"]).write(example_df_1)
        assert "p1=1" in os.listdir(tempdir), "Should partition with p1"

    with tempfile.TemporaryDirectory() as tempdir:
        Dataset(tempdir, partition_columns=["p1", "p2"]).write(example_df_1)
        assert "p1=1" in os.listdir(tempdir), "Should partition with a first"
        assert "p2=a" in os.listdir(f"{tempdir}/p1=1"), "Should partition with b second"


def test_read_partitions(example_ds_1: Dataset, example_df_1: pl.DataFrame) -> None:
    partitions = list(example_ds_1.read_partitions())
    assert (
        len(partitions) == 3
    ), "Three different combinations of a and b columns values"
    assert [p.shape for p in partitions] == [(1, 3), (1, 3), (2, 3)], "Should have read the right shapes"
    assert (
        pl.concat(partitions).to_dicts()
        == example_df_1.with_columns([pl.col("p1").cast(pl.Utf8)]).to_dicts()
    ), "Partition columns are read back as Utf8"


def test_partion_columns_should_be_string(example_ds_1: Dataset) -> None:
    df = example_ds_1.read()
    assert df.schema == {"v": pl.Int64, "p1": pl.Utf8, "p2": pl.Utf8}
    assert df.shape == (4,3)


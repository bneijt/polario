from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import pytest

from polario import unwrap
from polario.hive_dataset import HiveDataset, to_relative_location_from


def assert_equal(
    a: pl.DataFrame,
    b: pl.DataFrame,
    reason: str,
    partition_columns: list[str] | None = None,
) -> None:
    """Assert two dataframes are equal"""
    pcols = set(partition_columns or [])

    assert (
        [c for c in a.columns if c not in pcols]
        == [c for c in b.columns if c not in pcols]
    ), "Should have the same columns, in the same ordering, when ignoring partition columns"
    assert pcols - set(a.columns) == set(), "A should have all partition columns"
    assert pcols - set(b.columns) == set(), "B should have all partition columns"

    assert dict(a.schema.items()) == dict(
        b.schema.items()
    ), "Should have the same types"

    column_order = list(sorted(a.columns))

    def comparable_repr(df: pl.DataFrame) -> dict:
        return df.select_seq(column_order).sort(column_order).to_dict(as_series=False)

    assert comparable_repr(a) == comparable_repr(b), reason


@pytest.fixture
def example_df_1() -> pl.DataFrame:
    return pl.from_dicts(
        [
            {"p1": "1", "p2": "a", "v": 1},
            {"p1": "1", "p2": "b", "v": 1},
            {"p1": "2", "p2": "a", "v": 1},
            {"p1": "2", "p2": "a", "v": 2},
        ]
    )


@pytest.fixture
def example_ds_1(example_df_1: pl.DataFrame) -> Iterable[HiveDataset]:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = HiveDataset(
            tempdir, partition_columns=["p1", "p2"], max_rows_per_fragment=1
        )
        ds.write(example_df_1)
        yield ds


def test_dataset_should_raise_for_unsupported_protocol() -> None:
    with pytest.raises(ValueError):
        HiveDataset("example://some/url", partition_columns=[])


def test_write_should_raise_for_non_string_partition_columns(
    example_ds_1: HiveDataset,
) -> None:
    with pytest.raises(ValueError, match="Partition column p1 is not a string"):
        example_ds_1.write(pl.from_dicts([{"p1": 1, "p2": "a", "v": 1}]))


def test_write_partitioned_data(example_df_1: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        HiveDataset(tempdir).write(example_df_1)
        assert len(os.listdir(tempdir)) == 1, "Should write out one top-level fragment"

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


def test_use_relative_dataset_url_should_work(example_df_1: pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        old_wd = Path.cwd()
        try:
            os.chdir(tempdir)
            ds = HiveDataset(
                "target/banana", partition_columns=["p1"], max_rows_per_fragment=1
            )
            ds.write(example_df_1)
            ds = HiveDataset(
                "target/banana", partition_columns=["p1"], max_rows_per_fragment=1
            )
            unwrap(ds.scan()).collect()
        finally:
            os.chdir(old_wd)


def test_partion_columns_should_be_string(example_ds_1: HiveDataset) -> None:
    df = unwrap(example_ds_1.scan()).collect()
    assert df.schema == {"v": pl.Int64, "p1": pl.Utf8, "p2": pl.Utf8}
    assert df.shape == (4, 3)


def test_read_partion_should_read_single_partition(example_ds_1: HiveDataset) -> None:
    with pytest.raises(ValueError):
        # Missing p2
        example_ds_1.read_partition({"p1": "1"})
    assert example_ds_1.read_partition({"p1": "not_there", "p2": "not_there"}) is None

    partition_df = unwrap(example_ds_1.read_partition({"p1": "1", "p2": "a"}))
    assert partition_df.shape == (
        1,
        3,
    ), "This partition should contain only a single row and all columns"
    partition_df = unwrap(example_ds_1.read_partition({"p1": "2", "p2": "a"}))
    assert partition_df.shape == (
        2,
        3,
    ), "This partition should contain two rows and all columns"


def test_read_partition_should_raise_if_not_found(example_ds_1: HiveDataset) -> None:
    assert example_ds_1.read_partition({"p1": "not_there", "p2": "a"}) is None


def test_append_should_add_data(
    example_ds_1: HiveDataset, example_df_1: pl.DataFrame
) -> None:
    for _i in range(10):
        example_ds_1.append(example_df_1)
    assert (
        unwrap(example_ds_1.scan()).collect().shape[1] == example_df_1.shape[1]
    ), "Same number of columns"
    assert (
        unwrap(example_ds_1.scan()).collect().shape[0] == example_df_1.shape[0] * 11
    ), "Added ten times the original df in rows"


def test_should_support_schema_evolution() -> None:
    """If a partition has an extra column, the read_partitions should return a dataframe with that extra column"""
    row_a = {"p": "1", "a": "1", "b": 1}
    row_b = {"p": "2", "a": "1"}
    row_c = {"p": "1", "b": 2}
    with tempfile.TemporaryDirectory() as tempdir:
        ds = HiveDataset(tempdir, partition_columns=["p"], max_rows_per_fragment=1)
        ds.write(pl.from_dicts([row_a]))
        ds.write(pl.from_dicts([row_b]))
        partitions = list(ds.read_partitions())
        assert_equal(
            partitions[0],
            pl.from_dicts([row_a]),
            "Should read back all the data",
            ["p"],
        )
        assert_equal(
            partitions[1],
            pl.from_dicts([row_b]),
            "Should read back all the data",
            ["p"],
        )

        # Test append on the same partition
        ds.append(pl.from_dicts([row_c]))
        partitions = list(ds.read_partitions())
        assert_equal(
            partitions[0],
            pl.from_dicts([row_a, row_c]),
            "Should read back all the data",
            ["p"],
        )


def test__make_location_relative() -> None:
    burl = "s3://a/b"
    assert (
        to_relative_location_from("s3://", burl, "s3://a/b/some_type=1")
        == "some_type=1"
    )
    assert to_relative_location_from("s3://", burl, "/a/b/2021/01/01") == "2021/01/01"
    assert (
        to_relative_location_from("s3://", burl, "a/b/2021/01/01.hello")
        == "2021/01/01.hello"
    )
    assert to_relative_location_from("s3://", burl, "a/b/2021/01/") == "2021/01/"
    assert to_relative_location_from("s3://", burl, "a/b/a=1/b=2/") == "a=1/b=2/"
    assert (
        to_relative_location_from("s3://", burl, "s3://another/place/a=1")
        == "another/place/a=1"
    ), "Strip prefix if possible"
    assert (
        to_relative_location_from(
            "", "relative/to/wd", "/absolute/from/root/relative/to/wd/place/a=1"
        )
        == "place/a=1"
    ), "If base location is relative and location absolute, find based location in absolute location"
    assert (
        to_relative_location_from("", "relative/to/wd", "/relative/to/wd/place/a=1")
        == "place/a=1"
    ), "If base location is relative and location absolute, find based location in absolute location"

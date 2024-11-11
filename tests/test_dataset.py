import os
import tempfile
from typing import Type, Union

import polars as pl
import pytest

from polario import unwrap
from polario.delta_dataset import DeltaDataset
from polario.hive_dataset import HiveDataset


def assert_equal(
    a: pl.DataFrame,
    b: pl.DataFrame,
    reason: str,
) -> None:
    """Assert two dataframes are equal"""
    schema_a = {k: v for k, v in a.schema.items()}
    schema_b = {k: v for k, v in b.schema.items()}
    assert schema_a == schema_b, "Should have the same schema"
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


@pytest.mark.parametrize("Dataset", [HiveDataset, DeltaDataset])
def test_read_back_data(
    Dataset: Union[Type[HiveDataset], Type[DeltaDataset]], example_df_1: pl.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir)
        ds.write(example_df_1)
        assert len(os.listdir(tempdir)) >= 1, "Should write to disk"
        assert_equal(
            unwrap(ds.scan()).collect(), example_df_1, "Should read back the same data"
        )

    with tempfile.TemporaryDirectory() as tempdir:
        # Writing twice should result in the same data
        ds = Dataset(tempdir, partition_columns=["p1"])
        ds.write(example_df_1)
        ds.write(example_df_1)
        assert_equal(
            unwrap(ds.scan()).collect(), example_df_1, "Should read back the same data"
        )

    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1", "p2"])
        ds.write(example_df_1)
        assert_equal(
            unwrap(ds.scan()).collect(), example_df_1, "Should read back the same data"
        )


@pytest.mark.parametrize("Dataset", [HiveDataset, DeltaDataset])
def test_write_only_partitions_is_not_allowed(
    Dataset: Union[Type[HiveDataset], Type[DeltaDataset]],
) -> None:
    """Writing out dataframes that contain only partition columns is not allowed"""
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["a"])
        with pytest.raises(ValueError):
            ds.write(pl.from_dicts([{"a": "1"}]))
        with pytest.raises(ValueError):
            ds.append(pl.from_dicts([{"a": "1"}]))


@pytest.mark.parametrize("Dataset", [HiveDataset, DeltaDataset])
def test_scan_and_read_return_optional(
    Dataset: Union[Type[HiveDataset], Type[DeltaDataset]], example_df_1: pl.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1", "p2"])
        assert ds.scan() is None, "Should return None if no data is present"


@pytest.mark.parametrize("Dataset", [HiveDataset, DeltaDataset])
def test_read_partion_should_read_single_partition(
    Dataset: Union[Type[HiveDataset], Type[DeltaDataset]], example_df_1: pl.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1", "p2"])
        ds.write(example_df_1)
        with pytest.raises(
            ValueError,
        ):
            # Missing p2
            ds.read_partition({"p1": "1"})

        partition_df = unwrap(ds.read_partition({"p1": "1", "p2": "a"}))
        assert partition_df.shape == (
            1,
            3,
        ), "This partition should contain only a single row and all columns"
        partition_df = unwrap(ds.read_partition({"p1": "2", "p2": "a"}))
        assert partition_df.shape == (
            2,
            3,
        ), "This partition should contain two rows and all columns"


@pytest.mark.parametrize("Dataset", [HiveDataset])
def test_write_back_a_partition(
    Dataset: Type[HiveDataset], example_df_1: pl.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1"])
        ds.write(example_df_1)
        for partition in ds.read_partitions():
            ds.write(partition)
        assert_equal(
            unwrap(ds.scan()).collect(), example_df_1, "Should read back the same data"
        )


@pytest.mark.parametrize("Dataset", [HiveDataset, DeltaDataset])
def test_append_should_add_data(
    Dataset: Union[Type[HiveDataset], Type[DeltaDataset]], example_df_1: pl.DataFrame
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        ds = Dataset(tempdir, partition_columns=["p1", "p2"])
        for _i in range(10):
            ds.append(example_df_1)
        assert (
            unwrap(ds.scan()).collect().shape[1] == example_df_1.shape[1]
        ), "Same number of columns"
        assert (
            unwrap(ds.scan()).collect().shape[0] == example_df_1.shape[0] * 10
        ), "Added ten times the original df in rows"

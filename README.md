Polars IO utility library
=================

Helpers to make it easier to read and write Hive partitioned parquet dataset with Polars.

It is meant to be a library to deal with datasets easily, but also contains a commandline interface
which allows you to inspect parquet files and datasets more easily.

Dataset
=======
Example of use of `polario.dataset.HiveDataset`
```python

from polario.dataset import HiveDataset
import polars as pl
df = pl.from_dicts(
        [
            {"p1": 1, "v": 1},
            {"p1": 2, "v": 1},
        ]
    )

ds = HiveDataset("file:///tmp/", partition_columns=["p1"])

ds.write(df)

for partition_df in ds.read_partitions():
    print(partition_df)

```


To model data storage, we use three layers: dataset, partition, fragment.

Each dataset is a lexical ordered set of partitions
Each partition is a lexical ordered set of fragments
Each fragment is a file on disk with rows in any order

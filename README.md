Polars IO utility library
=================

Helpers to make it easier to read and write Hive partitioned parquet dataset with Polars.


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
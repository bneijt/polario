"""Main CLI interface
"""
import argparse
from pathlib import Path
from pprint import pprint

import polars as pl

from polario import __version__


def main() -> int:
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Polario: A Python package for polarimetric data analysis"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )
    parser.add_argument(
        "paths", metavar="PATH", type=Path, nargs="+", help="input paths"
    )
    args = parser.parse_args()

    for path in args.paths:
        if path.is_dir():
            raise ValueError(
                "Input path must be a file. File an issue if you want dataset support."
            )
        df = pl.read_parquet(path, use_pyarrow=True)
        print(df)

        if len(df):
            print("First row:")
            pprint(df.limit(1).to_dicts()[0])
        # print(type(path))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
